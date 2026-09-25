import type { CommonOptions, Manifest, RunInfo } from '../core/types.ts';
import { serialize, softmax } from '../core/util.ts';
import { loadTjs, repoOf } from './transformers.ts';

type Pipe = ((text: string | string[], o?: Record<string, unknown>) => Promise<unknown>) & {
  tokenizer: {
    (text: string, o?: Record<string, unknown>): { input_ids: { data: ArrayLike<number | bigint>; dims: number[] } };
    decode?: unknown;
    model?: { convert_ids_to_tokens?: (ids: number[]) => string[] };
    _tokenizer?: { encode: (t: string) => { tokens: string[]; ids: number[] } };
    do_lowercase_and_remove_accent?: boolean;
  };
  model: {
    config: { id2label?: Record<string, string>; label2id?: Record<string, number> };
    (inputs: unknown): Promise<{ logits: { data: Float32Array; dims: number[] } }>;
    dispose?: () => Promise<void>;
  };
  dispose?: () => Promise<void>;
};

const TASK: Record<string, string> = {
  'zero-shot-nli': 'zero-shot-classification',
  'sequence-classification': 'text-classification',
  'token-classification': 'token-classification',
};

export async function loadPipe(m: Manifest, opts: CommonOptions): Promise<{ pipe: Pipe; info: RunInfo }> {
  const r = await loadTjs<Pipe>(
    m,
    opts,
    async (t, base) => {
      const { repo } = repoOf(m);
      return (await t.pipeline(TASK[m.task] as never, repo, base as never)) as unknown as Pipe;
    },
    async (p) => {
      await p.dispose?.();
    },
  );
  return { pipe: r.value, info: r.info };
}

/** Probabilities over options, using natural-language inference. */
export async function nliChoice(
  m: Manifest,
  opts: CommonOptions,
  text: string,
  descriptions: string[],
  multiLabel = false,
): Promise<{ probs: number[]; info: RunInfo }> {
  const { pipe, info } = await loadPipe(m, opts);
  const template = (m.config?.hypothesis as string | undefined) ?? 'This example is {}.';
  const unique = [...new Set(descriptions)];
  const out = (await serialize(`judge:${info.model}`, () =>
    (pipe as unknown as (t: string, labels: string[], o: Record<string, unknown>) => Promise<unknown>)(text, unique, {
      hypothesis_template: template,
      multi_label: multiLabel,
    }),
  )) as { labels: string[]; scores: number[] };
  // The pipeline sorts labels by score; map back to the caller's order. Duplicate descriptions share a score.
  const byLabel = new Map<string, number>();
  out.labels.forEach((l, i) => byLabel.set(l, out.scores[i]));
  const probs = descriptions.map((d) => byLabel.get(d) ?? 0);
  if (!multiLabel) {
    const s = probs.reduce((a, b) => a + b, 0) || 1;
    return { probs: probs.map((p) => p / s), info };
  }
  return { probs, info };
}

/** Label probabilities from a sequence classifier, in id2label order. */
export async function classifyLabels(m: Manifest, opts: CommonOptions, text: string): Promise<{ labels: string[]; probs: number[]; info: RunInfo }> {
  const { pipe, info } = await loadPipe(m, opts);
  const enc = pipe.tokenizer(text, { truncation: true, max_length: (m.config?.maxLength as number | undefined) ?? 512 });
  const out = await serialize(`judge:${info.model}`, () => pipe.model(enc));
  const logits = Array.from(out.logits.data);
  const id2label = pipe.model.config.id2label ?? {};
  const labels = logits.map((_, i) => id2label[i] ?? `LABEL_${i}`);
  return { labels, probs: softmax(logits), info };
}

export interface RawSpan {
  type: string;
  start: number;
  end: number;
  text: string;
  score: number;
}

function cleanToken(tok: string): { core: string; continues: boolean } {
  if (tok.startsWith('##')) return { core: tok.slice(2), continues: true };
  if (tok.startsWith('▁')) return { core: tok.slice(1), continues: false };
  if (tok.startsWith('Ġ')) return { core: tok.slice(1), continues: false };
  return { core: tok, continues: false };
}

/**
 * Align tokenizer tokens to character offsets in the original text.
 * Works for WordPiece (##), SentencePiece (▁) and byte-level BPE (Ġ) vocabularies.
 */
export function alignTokens(text: string, tokens: string[], lowercase: boolean): ([number, number] | null)[] {
  const hay = lowercase ? text.toLowerCase() : text;
  let cursor = 0;
  return tokens.map((tok) => {
    const { core } = cleanToken(tok);
    if (!core || /^\[.*\]$|^<.*>$/.test(tok)) return null;
    const needle = lowercase ? core.toLowerCase() : core;
    const at = hay.indexOf(needle, cursor);
    if (at < 0 || at - cursor > 32) return null;
    cursor = at + needle.length;
    return [at, cursor];
  });
}

/** Character spans from a token-classification model, merged BIO-style. */
export async function tokenSpans(m: Manifest, opts: CommonOptions, text: string): Promise<{ spans: RawSpan[]; info: RunInfo }> {
  const { pipe, info } = await loadPipe(m, opts);
  const enc = pipe.tokenizer(text, { truncation: true, max_length: 512 });
  const out = await serialize(`judge:${info.model}`, () => pipe.model(enc));
  const [, seq, nLabels] = out.logits.dims;
  const ids = Array.from(enc.input_ids.data, (x) => Number(x));
  const raw = pipe.tokenizer._tokenizer?.encode(text);
  const tokens: string[] = raw && raw.tokens.length === ids.length ? raw.tokens : (pipe.tokenizer.model?.convert_ids_to_tokens?.(ids) ?? []);
  const lower = !!(pipe.tokenizer as unknown as { config?: { do_lower_case?: boolean } }).config?.do_lower_case;
  const offsets = alignTokens(text, tokens, lower);
  const id2label = pipe.model.config.id2label ?? {};
  const spans: RawSpan[] = [];
  let cur: (RawSpan & { n: number; sum: number }) | null = null;
  const close = () => {
    if (cur) spans.push({ type: cur.type, start: cur.start, end: cur.end, text: text.slice(cur.start, cur.end), score: cur.sum / cur.n });
    cur = null;
  };
  for (let i = 0; i < seq; i++) {
    const off = offsets[i];
    const row = out.logits.data.subarray(i * nLabels, (i + 1) * nLabels);
    const p = softmax(row);
    let best = 0;
    for (let k = 1; k < nLabels; k++) if (p[k] > p[best]) best = k;
    const lab = id2label[best] ?? 'O';
    if (!off) continue;
    if (lab === 'O') {
      close();
      continue;
    }
    const m2 = /^([BIES])-(.+)$/.exec(lab);
    const tag = m2 ? m2[1] : 'I';
    const type = m2 ? m2[2] : lab;
    const joined = cur && cur.type === type && tag !== 'B' && tag !== 'S' && off[0] - cur.end <= 1;
    const subword = cur && cleanToken(tokens[i]).continues && cur.end === off[0];
    if (cur && (joined || subword)) {
      cur.end = off[1];
      cur.n++;
      cur.sum += p[best];
    } else {
      close();
      cur = { type, start: off[0], end: off[1], text: '', score: 0, n: 1, sum: p[best] };
    }
  }
  close();
  return { spans, info };
}
