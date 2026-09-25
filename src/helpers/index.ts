/**
 * Task-named shortcuts. Each helper is one verb call you could write yourself.
 * @module
 */

import { toAudio } from '../backends/media.ts';
import { type VadOptions, detectSpeech as vadDetect } from '../backends/vad.ts';
import { ConfigError } from '../core/errors.ts';
import type { AudioLike, ImageLike } from '../core/parts.ts';
import { getPlatform } from '../core/runtime.ts';
import type { CommonOptions, ModelRef } from '../core/types.ts';
import { type GpuMode, scores as vectorScores } from '../gpu/vector.ts';
import { embed } from '../verbs/embed.ts';
import { choice, evaluate, label, type Question, type Span, spans } from '../verbs/evaluate.ts';
import { forecast } from '../verbs/forecast.ts';
import { type GenerateOptions, generate } from '../verbs/generate.ts';

type Opts = Partial<Omit<GenerateOptions, 'input' | 'messages'>>;

// ---------------------------------------------------------------- over generate

/** Speech to text. */
export function transcribe(audio: AudioLike, o: Opts = {}) {
  return generate({ model: 'stt:realtime', input: audio, ...o });
}

/** A caption for alt text. */
export function caption(image: ImageLike, o: Opts & { detail?: 'short' | 'detailed' | 'more-detailed' } = {}) {
  const { detail = 'detailed', ...rest } = o;
  const preset = detail === 'short' ? 'caption' : detail === 'detailed' ? 'caption-detailed' : 'caption-more-detailed';
  return generate({ model: 'vision:fast', input: image, preset, ...rest });
}

/** Text in an image, with regions. */
export function ocr(image: ImageLike, o: Opts = {}) {
  return generate({ model: 'vision:fast', input: image, preset: 'ocr', ...o });
}

/** Object boxes, or boxes for one phrase. */
export function detect(image: ImageLike, find?: string, o: Opts = {}) {
  return generate({ model: 'vision:fast', input: image, preset: find ? { find } : 'detect', ...o });
}

export interface WatchOptions extends Opts {
  prompt: string;
  /** Minimum time between frames. Default 1500 ms. */
  everyMs?: number;
  onResult: (r: { text: string; at: number }) => void;
  onError?: (err: unknown) => void;
}

/** Ask about a live video every few seconds. Skips frames while the model is busy. Returns stop(). */
export function watch(video: HTMLVideoElement, o: WatchOptions): () => void {
  const { prompt, everyMs = 1500, onResult, onError, model = 'vision:default', ...rest } = o;
  let stopped = false;
  const ac = new AbortController();
  (async () => {
    while (!stopped) {
      const started = Date.now();
      try {
        if (video.readyState >= 2) {
          const r = await generate({ model, input: [video, prompt], signal: ac.signal, ...rest });
          if (!stopped) onResult({ text: r.text, at: started });
        }
      } catch (err) {
        if (stopped) break;
        onError?.(err);
      }
      await new Promise((r) => setTimeout(r, Math.max(0, everyMs - (Date.now() - started))));
    }
  })();
  return () => {
    stopped = true;
    ac.abort();
  };
}

// ---------------------------------------------------------------- over evaluate

/** Pick the lane a prompt belongs to. */
export async function route<K extends string>(
  prompt: string,
  lanes: Record<K, string>,
  o: CommonOptions & { model?: ModelRef; threshold?: number; otherwise?: NoInfer<K> } = {},
) {
  const { model = 'judge:router', threshold, otherwise, ...common } = o;
  const r = await evaluate({ model, state: prompt, questions: { lane: choice(lanes, { threshold, otherwise }) }, ...common });
  return { route: r.answers.lane.choice, probabilities: r.answers.lane.probabilities, fellBack: r.answers.lane.fellBack, confidence: r.confidence.lane };
}

/** The label of a fixed-label classifier. */
export async function classify(text: string, model: ModelRef, o: CommonOptions = {}) {
  const r = await evaluate({ model, state: text, questions: { l: label() }, ...o });
  return r.answers.l;
}

export interface Finding {
  policy: string;
  answer: unknown;
}

/** Evaluate policies (questions with thresholds). `passed` is false when any answer is flagged. */
export async function lint(text: string, policies: Record<string, Question>, o: CommonOptions & { model?: ModelRef } = {}) {
  for (const [name, q] of Object.entries(policies)) {
    if (q.threshold === undefined && q.kind !== 'choice') throw new ConfigError(`Policy "${name}" needs a threshold.`);
  }
  const r = await evaluate({ state: text, questions: policies, ...o });
  const findings: Finding[] = Object.entries(r.answers)
    .filter(([, a]) => (a as { flagged?: boolean }).flagged)
    .map(([policy, answer]) => ({ policy, answer }));
  return { passed: findings.length === 0, findings, answers: r.answers };
}

/** Replace spans in text. Spans must not overlap; they are applied right to left. */
export function replaceSpans(text: string, found: Pick<Span, 'start' | 'end' | 'type'>[], mask: (s: Pick<Span, 'start' | 'end' | 'type'>) => string): string {
  let out = text;
  for (const s of [...found].sort((a, b) => b.start - a.start)) out = out.slice(0, s.start) + mask(s) + out.slice(s.end);
  return out;
}

/** Mask personal data. */
export async function redact(
  text: string,
  o: CommonOptions & { model?: ModelRef; types?: string[]; threshold?: number; mask?: (s: Pick<Span, 'start' | 'end' | 'type'>) => string } = {},
) {
  const { model = 'judge:pii', types, threshold = 0.5, mask = (s) => `[${s.type.toUpperCase()}]`, ...common } = o;
  const r = await evaluate({ model, state: text, questions: { pii: spans(types) }, ...common });
  return replaceSpans(
    text,
    r.answers.pii.spans.filter((s) => s.score >= threshold),
    mask,
  );
}

// ---------------------------------------------------------------- vectors

/** Cosine similarity. With normalized vectors this is the dot product. */
export function cosine(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new ConfigError(`Vectors have different lengths (${a.length} and ${b.length}).`);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na * nb) || 1);
}

export interface IndexItem<M = unknown> {
  id: string;
  vector: Float32Array;
  meta?: M;
}

export interface Hit<M = unknown> {
  id: string;
  score: number;
  meta?: M;
}

export interface VectorIndex<M = unknown> {
  readonly size: number;
  readonly dimensions: number;
  add(items: IndexItem<M> | IndexItem<M>[]): Promise<void>;
  remove(id: string): Promise<boolean>;
  search(query: Float32Array, o?: { k?: number; minScore?: number }): Promise<Hit<M>[]>;
  /** Near-duplicate pairs above a similarity threshold. */
  pairs(o?: { minScore?: number }): Promise<{ a: string; b: string; score: number }[]>;
  /** Embed a query with the index's model and search. */
  query(text: string, o?: { k?: number; minScore?: number }): Promise<Hit<M>[]>;
  /** Write the index to disk (server) or the Origin Private File System (browser). */
  save(): Promise<void>;
  clear(): Promise<void>;
}

export interface VectorIndexOptions {
  /** Name used for saving and loading. */
  name: string;
  /** Embedding model. Vectors from other models are refused. */
  model: ModelRef;
  dimensions?: number;
  /** Load a saved index with this name. Default true. */
  load?: boolean;
  /** Score on the GPU through vgpu. 'auto' uses it for large indexes on hardware GPUs. */
  gpu?: GpuMode;
}

async function storeBytes(name: string, data?: Uint8Array): Promise<Uint8Array | null> {
  const file = `edgewise-index-${name.replace(/[^\w.-]/g, '_')}.bin`;
  return getPlatform().store(file, data);
}

/** An exact k-NN index over normalized vectors, with persistence. Fast up to about 100,000 vectors. */
export async function vectorIndex<M = unknown>(o: VectorIndexOptions): Promise<VectorIndex<M>> {
  const modelTag = typeof o.model === 'string' ? o.model : o.model.id;
  let dims = o.dimensions ?? 0;
  let ids: string[] = [];
  let metas: (M | undefined)[] = [];
  let matrix = new Float32Array(0);
  if (o.load !== false) {
    const bytes = await storeBytes(o.name);
    if (bytes) {
      const headerLen = new DataView(bytes.buffer, bytes.byteOffset).getUint32(0, true);
      const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headerLen))) as { model: string; dims: number; ids: string[]; metas: M[] };
      if (header.model === modelTag && (!o.dimensions || header.dims === o.dimensions)) {
        dims = header.dims;
        ids = header.ids;
        metas = header.metas;
        const start = 4 + headerLen;
        matrix = new Float32Array(bytes.slice(start).buffer);
      }
    }
  }
  const norm = (v: Float32Array) => {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    const n = Math.sqrt(s) || 1;
    return v.map((x) => x / n);
  };
  const index: VectorIndex<M> = {
    get size() {
      return ids.length;
    },
    get dimensions() {
      return dims;
    },
    async add(items) {
      const list = Array.isArray(items) ? items : [items];
      if (!list.length) return;
      if (!dims) dims = list[0].vector.length;
      for (const it of list) {
        if (it.vector.length !== dims) throw new ConfigError(`Vector for "${it.id}" has ${it.vector.length} dimensions; the index uses ${dims}.`);
        const tag = (it as unknown as { model?: string }).model;
        if (tag && tag !== modelTag) throw new ConfigError(`Vector for "${it.id}" comes from "${tag}", but the index uses "${modelTag}".`);
      }
      const existing = new Map(ids.map((id, i) => [id, i]));
      const fresh = list.filter((it) => !existing.has(it.id));
      for (const it of list) {
        const at = existing.get(it.id);
        if (at !== undefined) {
          matrix.set(norm(it.vector), at * dims);
          metas[at] = it.meta;
        }
      }
      const next = new Float32Array(matrix.length + fresh.length * dims);
      next.set(matrix);
      fresh.forEach((it, i) => next.set(norm(it.vector), matrix.length + i * dims));
      matrix = next;
      ids = [...ids, ...fresh.map((f) => f.id)];
      metas = [...metas, ...fresh.map((f) => f.meta)];
    },
    async remove(id) {
      const at = ids.indexOf(id);
      if (at < 0) return false;
      const next = new Float32Array(matrix.length - dims);
      next.set(matrix.subarray(0, at * dims));
      next.set(matrix.subarray((at + 1) * dims), at * dims);
      matrix = next;
      ids.splice(at, 1);
      metas.splice(at, 1);
      return true;
    },
    async search(query, so = {}) {
      if (!ids.length) return [];
      if (query.length !== dims) throw new ConfigError(`Query has ${query.length} dimensions; the index uses ${dims}.`);
      const { scores } = await vectorScores(matrix, dims, norm(query), o.gpu);
      const k = Math.max(1, so.k ?? 10);
      const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
      const out: Hit<M>[] = [];
      for (const i of order) {
        if (out.length >= k) break;
        if (so.minScore !== undefined && scores[i] < so.minScore) break;
        out.push({ id: ids[i], score: scores[i], meta: metas[i] });
      }
      return out;
    },
    async pairs(po = {}) {
      const min = po.minScore ?? 0.92;
      const out: { a: string; b: string; score: number }[] = [];
      for (let i = 0; i < ids.length; i++) {
        const { scores } = await vectorScores(matrix, dims, matrix.subarray(i * dims, (i + 1) * dims), o.gpu);
        for (let j = i + 1; j < ids.length; j++) if (scores[j] >= min) out.push({ a: ids[i], b: ids[j], score: scores[j] });
      }
      return out.sort((x, y) => y.score - x.score);
    },
    async query(text, qo) {
      const { embedding } = await embed({ model: o.model, input: text, purpose: 'query', dimensions: o.dimensions });
      return index.search(embedding, qo);
    },
    async save() {
      const header = new TextEncoder().encode(JSON.stringify({ model: modelTag, dims, ids, metas }));
      const bytes = new Uint8Array(4 + header.length + matrix.byteLength);
      new DataView(bytes.buffer).setUint32(0, header.length, true);
      bytes.set(header, 4);
      bytes.set(new Uint8Array(matrix.buffer, matrix.byteOffset, matrix.byteLength), 4 + header.length);
      await storeBytes(o.name, bytes);
    },
    async clear() {
      ids = [];
      metas = [];
      matrix = new Float32Array(0);
    },
  };
  return index;
}

// ---------------------------------------------------------------- forecast and audio

export interface Anomaly {
  index: number;
  value: number;
  expected: [number, number];
  direction: 'high' | 'low';
}

/**
 * Flag values outside the range the model expected, using rolling one-step forecasts.
 * Batches the rolling windows so long series stay fast.
 */
export async function detectAnomalies(
  series: number[] | Float32Array,
  o: CommonOptions & {
    model?: ModelRef;
    /** Quantiles bounding the expected range. Default [0.1, 0.9]. */
    range?: [number, number];
    /** Extra margin on each side, as a fraction of the range's width. Default 0.5; 0 flags about 20% of noisy points. */
    tolerance?: number;
    /** Values used before the first check. Default 64. */
    warmup?: number;
    /** History used for each forecast. Default 512. */
    window?: number;
  } = {},
): Promise<Anomaly[]> {
  const { model = 'forecast:default', range = [0.1, 0.9], tolerance = 0.5, warmup = 64, window = 512, ...common } = o;
  const values = Array.from(series);
  if (values.length <= warmup) return [];
  const out: Anomaly[] = [];
  const batch = 16;
  for (let start = warmup; start < values.length; start += batch) {
    const idx = Array.from({ length: Math.min(batch, values.length - start) }, (_, i) => start + i);
    const contexts = idx.map((i) => values.slice(Math.max(0, i - window), i));
    const r = await forecast({ model, series: contexts, horizon: 1, quantiles: range, ...common });
    r.forecasts.forEach((f, j) => {
      const i = idx[j];
      const q0 = f.quantiles[range[0]][0];
      const q1 = f.quantiles[range[1]][0];
      const lo = q0 - tolerance * (q1 - q0);
      const hi = q1 + tolerance * (q1 - q0);
      const v = values[i];
      if (v < lo || v > hi) out.push({ index: i, value: v, expected: [lo, hi], direction: v > hi ? 'high' : 'low' });
    });
  }
  return out;
}

/** Find speech in recorded audio. Start and end are in seconds. */
export async function detectSpeech(audio: AudioLike, o: VadOptions & { sampleRate?: number } = {}) {
  const samples = await toAudio(audio, o.sampleRate);
  return vadDetect(samples, o);
}
