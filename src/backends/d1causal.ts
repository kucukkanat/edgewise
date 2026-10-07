import { ConfigError, UnsupportedInputError } from '../core/errors.ts';
import { dtypeLabel, getPlatform, loadCached, selectVariant } from '../core/runtime.ts';
import type { CommonOptions, Dtype, Manifest, RunInfo } from '../core/types.ts';
import { serialize, softmax } from '../core/util.ts';
import type { OrtModule, OrtSession } from '../platform/types.ts';
import { type D1Answer, type D1Question, IMAGE_PROCESSOR, packRows, resizeWeights, session } from './d1.ts';
import { fetchModelFile } from './files.ts';
import { getTransformers } from './transformers.ts';

/*
 * LiquidAI d1-3B: LFM2.5-VL-3B post-trained to answer at the last position of a chat prompt. Each option is
 * one token (yes/no, a letter, a digit), and the answer is a softmax over those tokens' logits. The prompt
 * and readout follow d1-3B's prompt.py; the graphs are described in scripts/export/d1_3b_export.py.
 */

type TJS = typeof import('@huggingface/transformers');
type RawImage = InstanceType<TJS['RawImage']>;
type Tok = ((t: string, o?: Record<string, unknown>) => { input_ids: { data: ArrayLike<number | bigint> } }) & {
  bos_token?: string;
  decode: (ids: number[], o?: Record<string, unknown>) => string;
};

const IM_START = '<|im_start|>';
const IM_END = '<|im_end|>';
const IMAGE = '<image>';
// Rows per run, capped by rows × longest row: the 3B decoder's activations grow fast with tokens.
const MAX_TOKENS = 8192;

/** A state as d1-3B reads it: text as itself, anything else as indented JSON (prompt.py's `json_only`). */
export function stateBlock(state: unknown): string {
  return typeof state === 'string' ? `${state}\n\n` : `${JSON.stringify(state, null, 2)}\n\n`;
}

/** Option codes: the labels when they already are single letters, else A–Z, else 00, 01, … */
export function optionCodes(labels: readonly string[]): string[] {
  const labs = labels.map((l) => l.trim());
  if (labs.length && labs.every((k) => k.length === 1 && /\p{L}/u.test(k))) return labs;
  if (labs.length <= 26) return labs.map((_, i) => String.fromCharCode(65 + i));
  return labs.map((_, i) => String(i).padStart(2, '0'));
}

const FALLBACK = [
  ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i)),
  ...Array.from({ length: 100 }, (_, i) => String(i).padStart(2, '0')),
  ...Array.from({ length: 26 }, (_, i) => String.fromCharCode(97 + i)),
  ...Array.from({ length: 200 }, (_, i) => `#${i}`),
  ...Array.from({ length: 26 * 26 }, (_, i) => String.fromCharCode(65 + Math.floor(i / 26)) + String.fromCharCode(65 + (i % 26))),
];

/** Each option's code and its single token, distinct across options (prompt.py's `aliases`). */
export function aliases(tokenize: (s: string) => number[], labels: readonly string[]): [string, number][] {
  const used = new Set<number>();
  const out: [string, number][] = [];
  const take = (raw: string) => {
    const enc = tokenize(raw);
    if (enc.length !== 1 || used.has(enc[0])) return false;
    out.push([raw, enc[0]]);
    used.add(enc[0]);
    return true;
  };
  for (const code of optionCodes(labels)) {
    if (!take(code) && !FALLBACK.some(take)) throw new ConfigError(`No single-token code is left for ${labels.length} options.`);
  }
  return out;
}

/** The single-token encodings among `texts`, without repeats (prompt.py's `_ids`). */
function singles(tokenize: (s: string) => number[], texts: readonly string[]): number[] {
  const out: number[] = [];
  for (const t of texts) {
    const enc = tokenize(t);
    if (enc.length === 1 && !out.includes(enc[0])) out.push(enc[0]);
  }
  return out;
}

/** The question turn, up to the answer slot. A noul's options read [yes, no]. */
export function questionBlock(tokenize: (s: string) => number[], q: D1Question): string {
  if (q.type === 'choice') {
    const codes = aliases(
      tokenize,
      q.options.map(([k]) => k),
    );
    const lines = q.options.map(([k, v], i) => `${codes[i][0]} ${v || k.replaceAll('_', ' ')}`).join('\n');
    return `${q.instructions}\n\nOptions:\n${lines}\n\nReply with the option code only.`;
  }
  if (q.type === 'noul') {
    // As prompt.py writes it, a side left out reads "None".
    const extra = q.true !== undefined || q.false !== undefined ? `\nYes: ${q.true ?? 'None'}\nNo: ${q.false ?? 'None'}` : '';
    return `${q.instructions}${extra}\n\nReply with yes or no only.`;
  }
  const legend = q.levels.map((l, i) => `${i} ${l}`).join('\n');
  return `${q.instructions}\n\n${legend}\n\nReply with a single digit 0-${q.levels.length - 1} only.`;
}

/** Token ids scored for each option, max-pooled (prompt.py's `readout_ids`). A noul reads [yes, no]. */
export function readoutIds(tokenize: (s: string) => number[], q: D1Question): number[][] {
  if (q.type === 'noul') {
    const groups = [singles(tokenize, ['yes', 'Yes', 'YES']), singles(tokenize, ['no', 'No', 'NO'])];
    if (groups.some((g) => !g.length)) throw new ConfigError('The tokenizer has no single-token yes or no.');
    return groups;
  }
  if (q.type === 'score') {
    const groups = q.levels.map((_, i) => singles(tokenize, [String(i)]));
    if (groups.some((g) => !g.length)) throw new ConfigError('score() needs single-token digits for every level.');
    return groups;
  }
  return aliases(
    tokenize,
    q.options.map(([k]) => k),
  ).map(([code, tid]) => [tid, ...singles(tokenize, [` ${code}`]).filter((i) => i !== tid)]);
}

/** The whole prompt for one question: optional images' markup, the state, the question and the assistant header. */
export function renderPrompt(tokenize: (s: string) => number[], bos: string, state: unknown, q: D1Question, images = ''): string {
  const body = state === null || state === undefined ? '' : `${stateBlock(state)}\nQUESTION:\n`;
  return `${bos}${IM_START}user\n${images}${body}${questionBlock(tokenize, q)}${IM_END}\n${IM_START}assistant\n`;
}

/** LFM2-VL's expansion of one `<image>`: its tiles with row/column markers, then the thumbnail. */
export function imageMarkup(rows: number, cols: number, thumbnailTokens: number, tileTokens = 256): string {
  const tiles = rows > 1 || cols > 1;
  let s = '<|image_start|>';
  if (tiles) {
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) s += `<|img_row_${r + 1}_col_${c + 1}|>${IMAGE.repeat(tileTokens)}`;
    s += `<|img_thumbnail|>${IMAGE.repeat(thumbnailTokens)}`;
  } else s += IMAGE.repeat(thumbnailTokens);
  return `${s}<|image_end|>`;
}

interface CausalConfig {
  tokenizer: { repo: string; revision: string };
  files: Record<'embed' | 'lower' | 'upper' | 'vision', Partial<Record<Dtype, string>>>;
  maxLength: number;
  imageTokenId: number;
  hidden: number;
}

interface Loaded {
  ort: OrtModule;
  tok: Tok;
  parts: Map<keyof CausalConfig['files'], Promise<OrtSession>>;
  device: string;
  dtype: Dtype;
}

function part(m: Manifest, l: Loaded, name: keyof CausalConfig['files'], opts: CommonOptions): Promise<OrtSession> {
  let s = l.parts.get(name);
  if (!s) {
    const files = (m.config as unknown as CausalConfig).files[name];
    const file = files[l.dtype] ?? Object.values(files)[0];
    if (!file) throw new ConfigError(`"${m.id}" has no ${name} file.`);
    s = fetchModelFile(m, file, { signal: opts.signal, onProgress: opts.onProgress }).then((b) => session(l.ort, b, l.device));
    s.catch(() => l.parts.delete(name));
    l.parts.set(name, s);
  }
  return s;
}

async function load(m: Manifest, opts: CommonOptions): Promise<{ value: Loaded; info: RunInfo }> {
  const sel = await selectVariant(m, opts);
  const c = m.config as unknown as CausalConfig;
  const dtype = (typeof sel.dtype === 'string' ? sel.dtype : 'q8') as Dtype;
  const value = await loadCached<Loaded>(
    `${m.id}|${sel.device}|${dtype}`,
    { manifest: m, selection: sel, progress: (e) => opts.onProgress?.(e), signal: opts.signal },
    async () => {
      const t = await getTransformers();
      const [tok, ort] = await Promise.all([t.AutoTokenizer.from_pretrained(c.tokenizer.repo, { revision: c.tokenizer.revision }), getPlatform().loadOrt()]);
      const l: Loaded = { ort, tok: tok as unknown as Tok, parts: new Map(), device: sel.device, dtype };
      // The decoder halves load with the model; vision loads on first use.
      await Promise.all([part(m, l, 'embed', opts), part(m, l, 'lower', opts), part(m, l, 'upper', opts)]);
      return l;
    },
    async (l) => {
      for (const s of l.parts.values()) await (await s.catch(() => null))?.release?.();
    },
  );
  const backend = getPlatform().isBrowser ? `onnxruntime-web:${sel.device}` : `onnxruntime-node:${sel.device}`;
  return { value, info: { model: m.id, device: sel.device, dtype: dtypeLabel(sel.dtype), backend } };
}

/** Every crop of the images through the vision graph, and the markup that stands in for each image. */
async function encodeImages(m: Manifest, l: Loaded, images: RawImage[], opts: CommonOptions): Promise<{ features: Float32Array[]; markup: string }> {
  const t = await getTransformers();
  const processor = new t.Lfm2VlImageProcessor(IMAGE_PROCESSOR);
  const vision = await part(m, l, 'vision', opts);
  const features: Float32Array[] = [];
  let markup = '';
  for (const image of images) {
    // d1-3B caps every picture at one megapixel before the processor.
    const scale = Math.sqrt((1024 * 1024) / (image.width * image.height));
    const capped =
      scale < 1 ? await image.resize(Math.max(1, Math.floor(image.width * scale)), Math.max(1, Math.floor(image.height * scale)), { resample: 3 }) : image;
    const r = (await processor(capped, { return_row_col_info: true })) as unknown as {
      pixel_values: { data: Float32Array; dims: number[] };
      spatial_shapes: { data: BigInt64Array };
      image_rows: number[];
      image_cols: number[];
      image_sizes: number[][];
    };
    const [crops, maxPatches, dim] = r.pixel_values.dims;
    for (let i = 0; i < crops; i++) {
      const h = Number(r.spatial_shapes.data[2 * i]);
      const w = Number(r.spatial_shapes.data[2 * i + 1]);
      const patches = r.pixel_values.data.subarray(i * maxPatches * dim, (i * maxPatches + h * w) * dim);
      const out = await serialize(`d1:${m.id}`, () =>
        vision.run({
          patches: new l.ort.Tensor('float32', patches, [1, h, w, dim]),
          rows: new l.ort.Tensor('float32', resizeWeights(h), [h, 16]),
          cols: new l.ort.Tensor('float32', resizeWeights(w), [w, 16]),
        }),
      );
      features.push(out.features.data as Float32Array);
    }
    const [th, tw] = r.image_sizes[0];
    const ds = (n: number) => Math.ceil(Math.floor(n / 16) / 2);
    markup += imageMarkup(r.image_rows[0], r.image_cols[0], ds(th) * ds(tw));
  }
  return { features, markup };
}

export interface D1CausalRequest {
  /** The state as given (a string, or any JSON value), or null when the images are the whole state. */
  state: unknown;
  questions: D1Question[];
  images?: RawImage[];
}

interface Row {
  ids: number[];
  /** Per position: 0 for a text token, else 1 + the row of `features` it takes. */
  image: number[];
  features: Float32Array[];
  question: D1Question;
  truncated: boolean;
}

/** Every request's answers, one per question, packed into as few passes as the token budget allows. */
export async function d1CausalDecide(m: Manifest, opts: CommonOptions, requests: D1CausalRequest[]): Promise<{ answers: D1Answer[][]; info: RunInfo }> {
  const { value: l, info } = await load(m, opts);
  const c = m.config as unknown as CausalConfig;
  const tokenize = (s: string) => Array.from(l.tok(s, { add_special_tokens: false }).input_ids.data, Number);
  const bos = l.tok.bos_token ?? '';
  const perRequest: Row[][] = [];
  for (const req of requests) {
    if (req.images?.length === 0 && (req.state === null || req.state === undefined)) throw new ConfigError('evaluate() needs a state or images.');
    const { features, markup } = req.images?.length ? await encodeImages(m, l, req.images, opts) : { features: [], markup: '' };
    perRequest.push(
      req.questions.map((question) => {
        let state = req.state;
        let ids = tokenize(renderPrompt(tokenize, bos, state, question, markup));
        let truncated = false;
        // Past the context, cut a text state from the right until the prompt fits. Tokens can merge across the
        // cut, so a few more are dropped than the overflow.
        if (ids.length > c.maxLength && typeof state === 'string') {
          const over = ids.length - c.maxLength;
          const stateIds = tokenize(state);
          if (over >= stateIds.length) throw new UnsupportedInputError(`The question and images alone exceed "${m.id}"'s ${c.maxLength} tokens.`);
          state = l.tok.decode(stateIds.slice(0, stateIds.length - over - 8));
          ids = tokenize(renderPrompt(tokenize, bos, state, question, markup));
          truncated = true;
        }
        if (ids.length > c.maxLength)
          throw new UnsupportedInputError(`The prompt is ${ids.length} tokens, more than "${m.id}"'s ${c.maxLength}.`, { hint: 'Send a shorter state.' });
        let k = 0;
        const image = ids.map((id) => (id === c.imageTokenId ? ++k : 0));
        return { ids, image, features, question, truncated };
      }),
    );
  }
  const probs: number[][] = [];
  for (const batch of packRows(perRequest.flat(), (r) => r.ids.length, MAX_TOKENS)) probs.push(...(await run(m, l, batch, tokenize)));
  let i = 0;
  return { answers: perRequest.map((rs) => rs.map((r) => ({ probs: probs[i++], truncated: r.truncated }))), info };
}

async function run(m: Manifest, l: Loaded, rows: Row[], tokenize: (s: string) => number[]): Promise<number[][]> {
  const c = m.config as unknown as CausalConfig;
  const [embed, lower, upper] = await Promise.all([part(m, l, 'embed', {}), part(m, l, 'lower', {}), part(m, l, 'upper', {})]);
  const B = rows.length;
  const T = Math.max(...rows.map((r) => r.ids.length));
  // Row 0 of the features is a placeholder, so the input is never empty. Rows sharing a request share its features.
  const blocks = [...new Set(rows.map((r) => r.features))];
  const offsets = new Map<Float32Array[], number>();
  let M = 1;
  for (const b of blocks) {
    offsets.set(b, M);
    M += b.reduce((n, f) => n + f.length / c.hidden, 0);
  }
  const features = new Float32Array(M * c.hidden);
  for (const b of blocks) {
    let at = (offsets.get(b) ?? 0) * c.hidden;
    for (const f of b) {
      features.set(f, at);
      at += f.length;
    }
  }
  const ids = new BigInt64Array(B * T);
  const index = new BigInt64Array(B * T);
  rows.forEach((r, b) => {
    const base = (offsets.get(r.features) ?? 1) - 1;
    r.ids.forEach((id, t) => {
      ids[b * T + t] = BigInt(id);
      if (r.image[t]) index[b * T + t] = BigInt(base + r.image[t]);
    });
  });
  const { ort } = l;
  const logits = await serialize(`d1:${m.id}`, async () => {
    const e = await embed.run({
      input_ids: new ort.Tensor('int64', ids, [B, T]),
      image_features: new ort.Tensor('float32', features, [M, c.hidden]),
      image_index: new ort.Tensor('int64', index, [B, T]),
    });
    const h = await lower.run({ embeds: e.embeds });
    const last = BigInt64Array.from(rows.map((r) => BigInt(r.ids.length - 1)));
    return (await upper.run({ hidden: h.hidden, last: new ort.Tensor('int64', last, [B]) })).logits.data as Float32Array;
  });
  const V = logits.length / B;
  return rows.map((r, b) => {
    const z = logits.subarray(b * V, (b + 1) * V);
    // Each option scores its best form; the answer is a softmax over those scores alone.
    const p = softmax(readoutIds(tokenize, r.question).map((g) => Math.max(...g.map((id) => z[id]))));
    // A noul reads [yes, no]; Edgewise's d1 answers give it as [false, true].
    return r.question.type === 'noul' ? p.reverse() : p;
  });
}
