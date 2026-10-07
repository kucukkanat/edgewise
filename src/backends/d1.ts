import { ConfigError, UnsupportedInputError } from '../core/errors.ts';
import { dtypeLabel, getPlatform, loadCached, selectVariant } from '../core/runtime.ts';
import type { CommonOptions, Dtype, Manifest, RunInfo } from '../core/types.ts';
import { serialize, softmax } from '../core/util.ts';
import type { OrtModule, OrtSession } from '../platform/types.ts';
import { fetchModelFile } from './files.ts';
import { getTransformers } from './transformers.ts';

type TJS = typeof import('@huggingface/transformers');
type RawImage = InstanceType<TJS['RawImage']>;
type Tok = (t: string, o?: Record<string, unknown>) => { input_ids: { data: ArrayLike<number | bigint> } };

/** A question in d1's terms. A noul is a yes/no question; its answer is P(yes). */
export type D1Question =
  | { type: 'choice'; instructions: string; options: readonly (readonly [string, string])[] }
  | { type: 'score'; instructions: string; levels: readonly string[] }
  | { type: 'noul'; instructions: string; true?: string; false?: string };

export type D1Media = 'text' | 'image' | 'audio';

// Delimiters from the tokenizer's reserved block (prompt.py's DELIM and MARKER).
const BOS = 1;
const MASK = 16;
const STATE = 17;
const QUESTION = 18;
const OPT = 19;
const OPT_END = 20;
const DECIDE = 21;
const QTYPE = { choice: 0, score: 1, noul: 2 } as const;
const HIDDEN = 1024;

/** `<|name|>` → `<¦name¦>`, so caller text can never forge a delimiter or the marker token. */
export function escapeDelimiters(text: string): string {
  return text.replace(/<\|([A-Za-z0-9_]+)\|>/g, '<¦$1¦>');
}

/** Python's `json.dumps(v, ensure_ascii=False)`, the serialization d1 was trained on. */
export function pyJson(v: unknown): string {
  if (v === null || v === undefined) return 'null';
  if (Array.isArray(v)) return `[${v.map(pyJson).join(', ')}]`;
  if (typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== undefined);
    return `{${entries.map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(', ')}}`;
  }
  return JSON.stringify(v);
}

/** The option texts in the model's order. A noul reads as [false, true]; audio questions were trained on plainer options. */
export function renderOptions(q: D1Question, media: D1Media): string[] {
  if (q.type === 'choice') {
    return q.options.map(([k, v], i) => (media === 'audio' ? `option_${String(i).padStart(3, '0')}: ${v || k}` : v ? `${k}: ${v}` : k));
  }
  if (q.type === 'score') return q.levels.map((l, i) => `level ${i}: ${l}`);
  if (media === 'audio' || (media === 'image' && q.true === undefined && q.false === undefined)) return ['false: no', 'true: yes'];
  return [`false: ${q.false || 'no, the statement does not hold'}`, `true: ${q.true || 'yes, the statement holds'}`];
}

/**
 * Token ids of one question over one state, and each option's marker position (prompt.py's `encode`):
 * `<bos> <state> state <q> instructions <opt> <mask> option </opt> … <decide>`. The options share a budget
 * of max(96, min(24k + 32, maxLen / 2)) tokens; the state is cut on the right to the room left.
 */
export function encodeQuestion(
  tokenize: (s: string) => number[],
  state: string,
  q: D1Question,
  maxLen: number,
  media: D1Media,
): { ids: number[]; markers: number[] } {
  const enc = (s: string) => tokenize(escapeDelimiters(s));
  const opts = renderOptions(q, media);
  const n = opts.length;
  const budget = Math.max(96, Math.min(n * 24 + 32, Math.floor(maxLen / 2)));
  const per = Math.max(2, Math.floor((budget - 3 * n) / n));
  const question = [QUESTION, ...enc(q.instructions)].slice(0, Math.max(16, budget));
  const markers: number[] = [];
  for (const text of opts) {
    markers.push(question.length + 1);
    question.push(OPT, MASK, ...enc(` ${text}`).slice(0, per), OPT_END);
  }
  question.push(DECIDE);
  const room = Math.max(0, maxLen - question.length - 2);
  const stateIds = [STATE, ...enc(state).slice(0, room)];
  const ids = [BOS, ...stateIds, ...question].slice(0, maxLen);
  const shifted = markers.map((m) => m + 1 + stateIds.length);
  if (shifted[shifted.length - 1] >= maxLen) throw new ConfigError('The options do not fit in the context.', { hint: 'Use fewer or shorter options.' });
  return { ids, markers: shifted };
}

/** The calibration temperature d1 learned for text questions of this type and option count. */
export function temperatureOf(temps: Record<string, number>, type: D1Question['type'], options: number): number {
  const band = options <= 2 ? '2' : options <= 5 ? '3-5' : options <= 10 ? '6-10' : '11+';
  return temps[`${type}:${band}`] ?? temps[type] ?? 1;
}

/** Valid conformer positions after the encoder's three stride-2 convolutions. */
export function subsampledLength(frames: number): number {
  let n = frames;
  for (let i = 0; i < 3; i++) n = Math.floor((n - 1) / 2) + 1;
  return n;
}

/**
 * The (out × size) matrix of PyTorch's antialiased bilinear resize along one axis (align_corners false), the
 * resize SigLIP2 applies to its position-embedding grid. Same weights as aten's `_compute_index_ranges_weights`.
 */
export function resizeWeights(out: number, size = 16): Float32Array {
  const scale = size / out;
  const support = scale >= 1 ? scale : 1;
  const inv = scale >= 1 ? 1 / scale : 1;
  const w = new Float32Array(out * size);
  for (let i = 0; i < out; i++) {
    const center = scale * (i + 0.5);
    const lo = Math.max(Math.trunc(center - support + 0.5), 0);
    const hi = Math.min(Math.trunc(center + support + 0.5), size);
    let total = 0;
    for (let j = lo; j < hi; j++) {
      const v = Math.max(0, 1 - Math.abs((j - center + 0.5) * inv));
      w[i * size + j] = v;
      total += v;
    }
    if (total > 0) for (let j = lo; j < hi; j++) w[i * size + j] /= total;
  }
  return w;
}

interface D1Config {
  tokenizer: { repo: string; revision: string };
  files: Record<'decide' | 'vision' | 'audio', Partial<Record<Dtype, string>>>;
  temperatures: Record<string, number>;
  maxLength: number;
  imageTextLength: number;
  audioTextLength: number;
}

interface Loaded {
  ort: OrtModule;
  tok: Tok;
  decide: OrtSession;
  // Vision and audio load on first use, inside the same cache entry, so they are evicted together.
  media: Map<'vision' | 'audio', Promise<OrtSession>>;
  device: string;
  dtype: Dtype;
}

// LFM2.5-VL-450M's tiling: up to ten 512 px tiles plus a thumbnail, 64 to 256 projected patches per crop.
const IMAGE_PROCESSOR = {
  do_resize: false,
  do_rescale: true,
  rescale_factor: 1 / 255,
  do_normalize: true,
  image_mean: [0.5, 0.5, 0.5],
  image_std: [0.5, 0.5, 0.5],
  downsample_factor: 2,
  encoder_patch_size: 16,
  tile_size: 512,
  min_tiles: 2,
  max_tiles: 10,
  use_thumbnail: true,
  min_image_tokens: 64,
  max_image_tokens: 256,
  max_pixels_tolerance: 2,
  do_image_splitting: true,
};
// NeMo's log-mel front end, as d1's audio encoder was trained.
const FEATURES = { feature_size: 128, sampling_rate: 16000, n_fft: 512, win_length: 400, hop_length: 160, preemphasis: 0.97 };
const MIN_SAMPLES = 8000;
const MAX_SAMPLES = 30 * 16000;

async function session(ort: OrtModule, bytes: Uint8Array | string, device: string): Promise<OrtSession> {
  const eps = device === 'webgpu' ? ['webgpu'] : getPlatform().isBrowser ? ['wasm'] : ['cpu'];
  return ort.InferenceSession.create(bytes, { executionProviders: eps });
}

function fileFor(m: Manifest, part: keyof D1Config['files'], dtype: Dtype): string {
  const files = (m.config as unknown as D1Config).files[part];
  const file = files[dtype] ?? Object.values(files)[0];
  if (!file) throw new ConfigError(`"${m.id}" has no ${part} file.`);
  return file;
}

async function loadD1(m: Manifest, opts: CommonOptions): Promise<{ value: Loaded; info: RunInfo }> {
  const sel = await selectVariant(m, opts);
  const c = m.config as unknown as D1Config;
  const dtype = (typeof sel.dtype === 'string' ? sel.dtype : 'q8') as Dtype;
  const value = await loadCached<Loaded>(
    `${m.id}|${sel.device}|${dtype}`,
    { manifest: m, selection: sel, progress: (e) => opts.onProgress?.(e), signal: opts.signal },
    async () => {
      const t = await getTransformers();
      const [bytes, tok, ort] = await Promise.all([
        fetchModelFile(m, fileFor(m, 'decide', dtype), { signal: opts.signal, onProgress: opts.onProgress }),
        t.AutoTokenizer.from_pretrained(c.tokenizer.repo, { revision: c.tokenizer.revision }),
        getPlatform().loadOrt(),
      ]);
      return { ort, tok: tok as unknown as Tok, decide: await session(ort, bytes, sel.device), media: new Map(), device: sel.device, dtype };
    },
    async (l) => {
      await l.decide.release?.();
      for (const s of l.media.values()) await (await s.catch(() => null))?.release?.();
    },
  );
  const backend = getPlatform().isBrowser ? `onnxruntime-web:${sel.device}` : `onnxruntime-node:${sel.device}`;
  return { value, info: { model: m.id, device: sel.device, dtype: dtypeLabel(sel.dtype), backend } };
}

function mediaSession(m: Manifest, l: Loaded, part: 'vision' | 'audio', opts: CommonOptions): Promise<OrtSession> {
  let s = l.media.get(part);
  if (!s) {
    s = fetchModelFile(m, fileFor(m, part, l.dtype), { signal: opts.signal, onProgress: opts.onProgress }).then((b) => session(l.ort, b, l.device));
    // A failed load is not cached, so the next call retries.
    s.catch(() => l.media.delete(part));
    l.media.set(part, s);
  }
  return s;
}

/** Images → (P, 1024) prefix embeddings: every image's tiles, then its thumbnail, images in order. */
async function imagePrefix(m: Manifest, l: Loaded, images: RawImage[], opts: CommonOptions): Promise<Float32Array> {
  const t = await getTransformers();
  const processor = new t.Lfm2VlImageProcessor(IMAGE_PROCESSOR);
  const { pixel_values, spatial_shapes } = (await processor(images)) as {
    pixel_values: { data: Float32Array; dims: number[] };
    spatial_shapes: { data: BigInt64Array };
  };
  const s = await mediaSession(m, l, 'vision', opts);
  const [crops, maxPatches, dim] = pixel_values.dims;
  const parts: Float32Array[] = [];
  // One crop per run: the graph reads h and w from the patch grid's shape (see scripts/export/d1_export.py).
  for (let i = 0; i < crops; i++) {
    const h = Number(spatial_shapes.data[2 * i]);
    const w = Number(spatial_shapes.data[2 * i + 1]);
    const patches = pixel_values.data.subarray(i * maxPatches * dim, (i * maxPatches + h * w) * dim);
    const out = await serialize(`d1:${m.id}`, () =>
      s.run({
        patches: new l.ort.Tensor('float32', patches, [1, h, w, dim]),
        rows: new l.ort.Tensor('float32', resizeWeights(h), [h, 16]),
        cols: new l.ort.Tensor('float32', resizeWeights(w), [w, 16]),
      }),
    );
    parts.push(out.features.data as Float32Array);
  }
  const prefix = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((off, p) => (prefix.set(p, off), off + p.length), 0);
  return prefix;
}

/** 16 kHz mono samples → (P, 1024) prefix embeddings, one per 80 ms. */
async function audioPrefix(m: Manifest, l: Loaded, samples: Float32Array, opts: CommonOptions): Promise<Float32Array> {
  const t = await getTransformers();
  let x = samples.subarray(0, MAX_SAMPLES);
  if (x.length < MIN_SAMPLES) {
    const padded = new Float32Array(MIN_SAMPLES);
    padded.set(x);
    x = padded;
  }
  const fe = new t.ParakeetFeatureExtractor(FEATURES);
  const { input_features } = (await fe(x)) as { input_features: { data: Float32Array; dims: number[] } };
  const frames = Math.floor(x.length / FEATURES.hop_length);
  const s = await mediaSession(m, l, 'audio', opts);
  const out = await serialize(`d1:${m.id}`, () =>
    s.run({
      input_features: new l.ort.Tensor('float32', input_features.data, input_features.dims),
      length: new l.ort.Tensor('int64', BigInt64Array.of(BigInt(frames)), [1]),
    }),
  );
  return (out.prefix.data as Float32Array).slice(0, subsampledLength(frames) * HIDDEN);
}

export interface D1Request {
  /** The state as text (already serialized). */
  state: string;
  questions: D1Question[];
  images?: RawImage[];
  /** 16 kHz mono samples. */
  audio?: Float32Array;
}

/**
 * Each question's distribution over its options, in option order (a noul as [false, true]),
 * from one batched forward pass. Media are encoded once and shared by every question.
 */
export async function d1Decide(m: Manifest, opts: CommonOptions, req: D1Request): Promise<{ probs: number[][]; info: RunInfo }> {
  if (req.images?.length && req.audio) throw new UnsupportedInputError(`"${m.id}" takes images or audio in one call, not both.`);
  const { value: l, info } = await loadD1(m, opts);
  const c = m.config as unknown as D1Config;
  const media: D1Media = req.images?.length ? 'image' : req.audio ? 'audio' : 'text';
  const prefix =
    media === 'image' && req.images
      ? await imagePrefix(m, l, req.images, opts)
      : media === 'audio' && req.audio
        ? await audioPrefix(m, l, req.audio, opts)
        : null;
  const P = prefix ? prefix.length / HIDDEN : 0;
  const maxLen = Math.min(media === 'image' ? c.imageTextLength : media === 'audio' ? c.audioTextLength : c.maxLength, c.maxLength - P);
  if (maxLen < 64) {
    throw new UnsupportedInputError(`The images take ${P} of the ${c.maxLength} positions.`, { hint: 'Send fewer or smaller images.' });
  }
  const tokenize = (s: string) => Array.from(l.tok(s, { add_special_tokens: false }).input_ids.data, Number);
  const rows = req.questions.map((q) => encodeQuestion(tokenize, req.state, q, maxLen, media));
  const B = rows.length;
  const T = Math.max(...rows.map((r) => r.ids.length));
  const K = Math.max(...rows.map((r) => r.markers.length));
  const ids = new BigInt64Array(B * T);
  const mask = new BigInt64Array(B * T);
  const markerPos = new BigInt64Array(B * K);
  const markerMask = new BigInt64Array(B * K);
  rows.forEach((r, b) => {
    r.ids.forEach((id, i) => {
      ids[b * T + i] = BigInt(id);
      mask[b * T + i] = 1n;
    });
    r.markers.forEach((p, k) => {
      markerPos[b * K + k] = BigInt(p);
      markerMask[b * K + k] = 1n;
    });
  });
  // Text-only calls pass one masked-out prefix position; see scripts/export/d1_export.py.
  const Pin = Math.max(P, 1);
  const prefixData = new Float32Array(B * Pin * HIDDEN);
  if (prefix) for (let b = 0; b < B; b++) prefixData.set(prefix, b * Pin * HIDDEN);
  const { ort } = l;
  const out = await serialize(`d1:${m.id}`, () =>
    l.decide.run({
      input_ids: new ort.Tensor('int64', ids, [B, T]),
      attention_mask: new ort.Tensor('int64', mask, [B, T]),
      prefix: new ort.Tensor('float32', prefixData, [B, Pin, HIDDEN]),
      prefix_mask: new ort.Tensor('int64', new BigInt64Array(B * Pin).fill(prefix ? 1n : 0n), [B, Pin]),
      marker_pos: new ort.Tensor('int64', markerPos, [B, K]),
      marker_mask: new ort.Tensor('int64', markerMask, [B, K]),
      qtype: new ort.Tensor('int64', BigInt64Array.from(req.questions.map((q) => BigInt(QTYPE[q.type]))), [B]),
    }),
  );
  const logits = out.logits.data as Float32Array;
  const probs = req.questions.map((q, b) => {
    const n = rows[b].markers.length;
    // Learned temperatures calibrate text answers; image and audio answers use the softmax as trained.
    const temp = media === 'text' ? temperatureOf(c.temperatures, q.type, n) : 1;
    return softmax(Array.from(logits.subarray(b * K, b * K + n), (z) => z / temp));
  });
  return { probs, info };
}
