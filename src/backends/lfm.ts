import { getConfig } from '../core/config.ts';
import { dtypeLabel, getPlatform, loadCached, selectVariant } from '../core/runtime.ts';
import type { CommonOptions, Dtype, Manifest, RunInfo } from '../core/types.ts';
import { serialize, softmax } from '../core/util.ts';
import type { OrtModule, OrtSession } from '../platform/types.ts';
import { fetchModelFile } from './files.ts';
import type { RawSpan } from './judge.ts';
import { getTransformers } from './transformers.ts';

type Tok = ((t: string, o?: Record<string, unknown>) => { input_ids: { data: ArrayLike<number | bigint> } }) & {
  decode: (ids: number[], o?: Record<string, unknown>) => string;
};

interface Loaded {
  ort: OrtModule;
  session: OrtSession;
  tok: Tok;
  id2label?: Record<string, string>;
}

interface LfmConfig {
  tokenizer: { repo: string; revision: string };
  files: Partial<Record<Dtype, string>>;
  parts?: Partial<Record<Dtype, number>>;
  bos?: number;
  id2label?: Record<string, string>;
}

export async function loadLfm(m: Manifest, opts: CommonOptions): Promise<{ value: Loaded; info: RunInfo }> {
  const sel = await selectVariant(m, opts);
  const c = m.config as unknown as LfmConfig;
  const dtype = (typeof sel.dtype === 'string' ? sel.dtype : 'q8') as Dtype;
  const file = c.files[dtype] ?? Object.values(c.files)[0] ?? 'onnx/model.onnx';
  const value = await loadCached<Loaded>(
    `${m.id}|${sel.device}|${dtype}`,
    { manifest: m, selection: sel, progress: (e) => opts.onProgress?.(e), signal: opts.signal },
    async () => {
      const t = await getTransformers();
      const [bytes, tok] = await Promise.all([
        fetchModelFile(m, file, { parts: c.parts?.[dtype], signal: opts.signal, onProgress: opts.onProgress }),
        t.AutoTokenizer.from_pretrained(c.tokenizer.repo, { revision: c.tokenizer.revision }),
      ]);
      const ort = await getPlatform().loadOrt();
      const eps = sel.device === 'webgpu' ? ['webgpu'] : getPlatform().isBrowser ? ['wasm'] : ['cpu'];
      const session = await ort.InferenceSession.create(bytes, { executionProviders: eps });
      let id2label = c.id2label;
      if (!id2label && m.task === 'lfm-token-classification') {
        const url = `${getConfig().hub.replace(/\/$/, '')}/${c.tokenizer.repo}/resolve/${c.tokenizer.revision}/config.json`;
        const json = JSON.parse(new TextDecoder().decode(await getPlatform().fetchCached(url, { signal: opts.signal })));
        id2label = json.id2label as Record<string, string>;
      }
      return { ort, session, tok: tok as unknown as Tok, id2label };
    },
    async (l) => {
      await l.session.release?.();
    },
  );
  const backend = getPlatform().isBrowser ? `onnxruntime-web:${sel.device}` : `onnxruntime-node:${sel.device}`;
  return { value, info: { model: m.id, device: sel.device, dtype: dtypeLabel(sel.dtype), backend } };
}

function ids(tok: Tok, s: string): number[] {
  return Array.from(tok(s, { add_special_tokens: false }).input_ids.data, (x) => Number(x));
}

/**
 * Score text against lanes with the LFM2.5 prompt router, in one encoder pass.
 * Inputs are tokenized piece by piece so category and text tokens are known exactly.
 */
export async function lfmRoute(m: Manifest, opts: CommonOptions, text: string, lanes: string[]): Promise<{ probs: number[]; info: RunInfo }> {
  const { value, info } = await loadLfm(m, opts);
  const { tok, ort, session } = value;
  const c = m.config as unknown as LfmConfig;
  const all: number[] = [c.bos ?? 1];
  const kind: number[] = [-2];
  const push = (s: string, k: number) => {
    for (const id of ids(tok, s)) {
      all.push(id);
      kind.push(k);
    }
  };
  push('Categories:\n', -2);
  lanes.forEach((lane, i) => {
    push('- ', -2);
    push(lane, i);
    push('\n', -2);
  });
  push('\nText:\n', -2);
  push(text, -1);
  const S = all.length;
  const R = lanes.length;
  const textIdx = kind.map((k, i) => (k === -1 ? i : -1)).filter((i) => i >= 0);
  const textPool = new Float32Array(S);
  for (const i of textIdx) textPool[i] = 1 / textIdx.length;
  const catPool = new Float32Array(R * S);
  for (let r = 0; r < R; r++) {
    const idx = kind.map((k, i) => (k === r ? i : -1)).filter((i) => i >= 0);
    for (const i of idx) catPool[r * S + i] = 1 / idx.length;
  }
  const out = await serialize(`lfm:${m.id}`, () =>
    session.run({
      input_ids: new ort.Tensor('int64', BigInt64Array.from(all.map(BigInt)), [1, S]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(S).fill(1n), [1, S]),
      text_pool: new ort.Tensor('float32', textPool, [1, 1, S]),
      category_pool: new ort.Tensor('float32', catPool, [1, R, S]),
    }),
  );
  return { probs: softmax(Array.from(out.logits.data as Float32Array)), info };
}

/** Character offsets of each token, found by decoding growing prefixes (exact for byte-level BPE). */
export function decodeOffsets(decode: (ids: number[]) => string, tokenIds: number[], text: string): ([number, number] | null)[] {
  const out: ([number, number] | null)[] = [];
  let prevLen = 0;
  let cursor = 0;
  for (let i = 0; i < tokenIds.length; i++) {
    const s = decode(tokenIds.slice(0, i + 1));
    const piece = s.slice(prevLen);
    prevLen = s.length;
    const core = piece.replace(/^\s+/, '');
    if (!core || core.includes('�')) {
      out.push(null);
      continue;
    }
    const at = text.indexOf(core, cursor);
    if (at < 0) {
      out.push(null);
      continue;
    }
    cursor = at + core.length;
    out.push([at, cursor]);
  }
  return out;
}

/** PII spans from the LFM2.5 detector, decoded from its BIES tags. */
export async function lfmSpans(m: Manifest, opts: CommonOptions, text: string): Promise<{ spans: RawSpan[]; info: RunInfo }> {
  const { value, info } = await loadLfm(m, opts);
  const { tok, ort, session } = value;
  const c = m.config as unknown as LfmConfig;
  const body = ids(tok, text).slice(0, 8190);
  const all = [c.bos ?? 1, ...body];
  const S = all.length;
  const out = await serialize(`lfm:${m.id}`, () =>
    session.run({
      input_ids: new ort.Tensor('int64', BigInt64Array.from(all.map(BigInt)), [1, S]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(S).fill(1n), [1, S]),
    }),
  );
  const logits = out.logits.data as Float32Array;
  const nLabels = logits.length / S;
  const offsets = [null, ...decodeOffsets((x) => tok.decode(x, { skip_special_tokens: true }), body, text)];
  const id2label = value.id2label ?? c.id2label ?? {};
  const spans: RawSpan[] = [];
  let cur: { type: string; start: number; end: number; sum: number; n: number } | null = null;
  const close = () => {
    if (cur) spans.push({ type: cur.type, start: cur.start, end: cur.end, text: text.slice(cur.start, cur.end), score: cur.sum / cur.n });
    cur = null;
  };
  for (let i = 0; i < S; i++) {
    const off = offsets[i];
    if (!off) continue;
    const p = softmax(logits.subarray(i * nLabels, (i + 1) * nLabels));
    let best = 0;
    for (let k = 1; k < nLabels; k++) if (p[k] > p[best]) best = k;
    const lab = id2label[String(best)] ?? 'O';
    if (lab === 'O') {
      close();
      continue;
    }
    const mm = /^([BIES])-(.+)$/.exec(lab);
    const tag = mm ? mm[1] : 'I';
    const type = mm ? mm[2] : lab;
    if (cur && cur.type === type && (tag === 'I' || tag === 'E')) {
      cur.end = off[1];
      cur.sum += p[best];
      cur.n++;
    } else {
      close();
      cur = { type, start: off[0], end: off[1], sum: p[best], n: 1 };
    }
    if (tag === 'E' || tag === 'S') close();
  }
  close();
  return { spans, info };
}
