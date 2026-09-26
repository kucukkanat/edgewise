/**
 * Run Edgewise in a Web Worker so model loading and inference never block the page.
 *
 * In a worker file:
 * ```ts
 * import { serveWorker } from 'edgewise/worker';
 * serveWorker();
 * ```
 * On the page:
 * ```ts
 * import { connectWorker } from 'edgewise/worker';
 * const ew = connectWorker(new Worker(new URL('./edgewise.worker.ts', import.meta.url), { type: 'module' }));
 * for await (const delta of ew.generate({ model: 'text:default', input: 'Hi' })) out.append(delta);
 * ```
 * The page-side object has the same verbs and options as the main package. Streams, events, cancel,
 * progress, tools (executed on the page), schemas (checked on the page), microphone sources and
 * streamed text for speak() all cross the worker boundary.
 * @module
 */
import {
  AbortError,
  BackendError,
  ConfigError,
  DownloadError,
  EdgewiseError,
  ModelNotFoundError,
  OutOfMemoryError,
  PermissionError,
  SchemaValidationError,
  StorageQuotaError,
  UnsupportedDeviceError,
  UnsupportedInputError,
  WrongVerbError,
} from '../core/errors.ts';
import type { AudioSource, ImageLike, RawPixels } from '../core/parts.ts';
import { registry } from '../core/registry.ts';
import { Run, type RunContext, type RunEvent } from '../core/run.ts';
import type { Capabilities, LoadEvent, Manifest } from '../core/types.ts';
import { isAsyncIterable } from '../core/util.ts';
import * as ew from '../index.ts';
import type { EmbedManyOptions, EmbedManyResult, EmbedOneOptions, EmbedResult } from '../verbs/embed.ts';
import type { EvaluateOptions, EvaluateResult, Question } from '../verbs/evaluate.ts';
import type { ForecastManyResult, ForecastOptions, ForecastResult, SeriesInput } from '../verbs/forecast.ts';
import type { GenerateChunk, GenerateOptions, GenerateResult } from '../verbs/generate.ts';
import { imageFromRgba, type PaintOptions, type PaintResult, type PaintStep } from '../verbs/paint.ts';
import { toJsonSchema, validateWith } from '../verbs/schema.ts';
import {
  audioFromSamples,
  type ClonedVoice,
  type CloneVoiceOptions,
  type SpeakOptions,
  SpeakRun,
  type SpeechAudio,
  type SpeechChunk,
  type Voice,
} from '../verbs/speak.ts';
import type { AnyTool } from '../verbs/tools.ts';

/* ---------------------------------------------------------------- protocol */

type Op =
  | 'generate'
  | 'evaluate'
  | 'embed'
  | 'speak'
  | 'cloneVoice'
  | 'listVoices'
  | 'paint'
  | 'forecast'
  | 'preload'
  | 'unload'
  | 'capabilities'
  | 'configure'
  | 'models';

type ToMain =
  | { id: number; type: 'chunk'; chunk: unknown }
  | { id: number; type: 'event'; event: RunEvent }
  | { id: number; type: 'result'; result: unknown }
  | { id: number; type: 'error'; error: SerializedError }
  | { id: number; type: 'tool'; call: number; name: string; input: unknown }
  | { id: number; type: 'approve'; call: number; request: unknown }
  | { id: number; type: 'batch'; done: number; total: number }
  | { type: 'stream-cancel'; stream: number };

type ToWorker =
  | {
      id: number;
      type: 'call';
      op: Op;
      args: unknown;
      tools?: Record<string, { description: string; inputSchema: unknown; execute: boolean }>;
      approve?: boolean;
    }
  | { id: number; type: 'cancel' }
  | { id: number; type: 'tool-result'; call: number; output?: unknown; error?: string }
  | { id: number; type: 'approve-result'; call: number; ok: boolean }
  | { type: 'stream'; stream: number; value?: unknown; done?: boolean; error?: string };

interface SerializedError {
  name: string;
  code?: string;
  message: string;
  hint?: string;
  retryable?: boolean;
  raw?: string;
  issues?: unknown;
}

interface Port {
  postMessage(m: unknown, transfer?: Transferable[]): void;
  addEventListener(type: 'message', fn: (e: MessageEvent) => void): void;
}

const MARK = '__edgewise';

/* ---------------------------------------------------------------- page side */

/** The verbs, proxied to a worker. Same options and results as the main package. */
export interface EdgewiseWorker {
  generate<T = unknown>(options: GenerateOptions<T>): Run<GenerateResult<T>, GenerateChunk<T>>;
  evaluate<Q extends Record<string, Question>>(options: EvaluateOptions<Q>): Promise<EvaluateResult<Q>>;
  embed(options: EmbedOneOptions): Promise<EmbedResult>;
  embed(options: EmbedManyOptions): Promise<EmbedManyResult>;
  speak(options: SpeakOptions): SpeakRun;
  cloneVoice(options: CloneVoiceOptions): Promise<ClonedVoice>;
  listVoices(model?: string): Promise<Voice[]>;
  paint(options: PaintOptions): Run<PaintResult, PaintStep>;
  forecast(options: ForecastOptions & { series: SeriesInput }): Promise<ForecastResult>;
  forecast(options: ForecastOptions & { series: SeriesInput[] }): Promise<ForecastManyResult>;
  preload(models: string | string[], o?: { onProgress?: (e: LoadEvent) => void; allowPreview?: boolean }): Promise<void>;
  unload(id?: string): Promise<void>;
  capabilities(): Promise<Capabilities>;
  /** Apply configure() inside the worker. */
  configure(options: Record<string, unknown>): Promise<void>;
  /** Model IDs registered in the worker. */
  models(): Promise<string[]>;
  /** Stop the worker. Pending runs reject with AbortError. */
  terminate(): void;
}

const ERRORS: Record<string, new (message: string, o: never) => EdgewiseError> = {
  E_INPUT: UnsupportedInputError,
  E_UNSUPPORTED: UnsupportedDeviceError,
  E_MODEL: ModelNotFoundError,
  E_VERB: WrongVerbError,
  E_DOWNLOAD: DownloadError,
  E_QUOTA: StorageQuotaError,
  E_OOM: OutOfMemoryError,
  E_BACKEND: BackendError,
  E_SCHEMA: SchemaValidationError as never,
  E_PERMISSION: PermissionError,
  E_ABORT: AbortError,
  E_CONFIG: ConfigError,
};

/** Rebuild an error with its original class, so `instanceof` works across the worker boundary. */
function reviveError(e: SerializedError): Error {
  const Cls = e.code ? ERRORS[e.code] : undefined;
  if (!Cls) {
    const err = new Error(e.message);
    err.name = e.name;
    return err;
  }
  return new Cls(e.message, { hint: e.hint, retryable: e.retryable, raw: e.raw ?? '', issues: e.issues } as never);
}

function hasFunction(v: unknown, path = ''): string | null {
  if (typeof v === 'function') return path || '(root)';
  if (v === null || typeof v !== 'object' || ArrayBuffer.isView(v)) return null;
  for (const [k, x] of Object.entries(v)) {
    const f = hasFunction(x, path ? `${path}.${k}` : k);
    if (f) return f;
  }
  return null;
}

const isDom = (v: unknown, ctor: string) => {
  const C = (globalThis as Record<string, unknown>)[ctor];
  return typeof C === 'function' && v instanceof (C as new (...a: never[]) => unknown);
};

function pixelsOf(src: CanvasImageSource, w: number, h: number): RawPixels {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d') as OffscreenCanvasRenderingContext2D;
  ctx.drawImage(src, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h);
  return { data: d.data, width: w, height: h, channels: 4 };
}

async function pageImage(v: ImageLike): Promise<unknown> {
  if (isDom(v, 'HTMLImageElement')) {
    const el = v as HTMLImageElement;
    if (!el.complete) await el.decode();
    return pixelsOf(el, el.naturalWidth, el.naturalHeight);
  }
  if (isDom(v, 'HTMLVideoElement')) {
    const el = v as HTMLVideoElement;
    if (!el.videoWidth) throw new UnsupportedInputError('The video element has no frame yet.', { hint: 'Wait for the video to start playing.' });
    return pixelsOf(el, el.videoWidth, el.videoHeight);
  }
  if (isDom(v, 'HTMLCanvasElement') || isDom(v, 'OffscreenCanvas')) {
    const c = v as HTMLCanvasElement;
    return pixelsOf(c, c.width, c.height);
  }
  return v;
}

/** Page-side conversion: DOM media to pixels, URLs to markers, live sources to streams. */
async function toWire(v: unknown, streams: (src: AsyncIterable<unknown>) => number, key?: string): Promise<unknown> {
  if (v === null || typeof v !== 'object') return typeof v === 'function' ? undefined : v;
  if (v instanceof URL) return { [MARK]: 'url', href: v.href };
  if (isDom(v, 'HTMLImageElement') || isDom(v, 'HTMLCanvasElement') || isDom(v, 'OffscreenCanvas')) return pageImage(v as ImageLike);
  if (isDom(v, 'HTMLVideoElement')) {
    if (key === 'video') {
      const { sampleVideoFrames } = await import('../backends/lm.ts');
      return sampleVideoFrames(v as HTMLVideoElement, 8);
    }
    return pageImage(v as ImageLike);
  }
  if (isDom(v, 'AudioBuffer')) {
    const b = v as AudioBuffer;
    const out = new Float32Array(b.length);
    for (let c = 0; c < b.numberOfChannels; c++) {
      const d = b.getChannelData(c);
      for (let i = 0; i < d.length; i++) out[i] += d[i] / b.numberOfChannels;
    }
    return { [MARK]: 'pcm', samples: out, sampleRate: b.sampleRate };
  }
  if ((v as AudioSource).kind === 'audio-source') {
    const s = v as AudioSource;
    return { [MARK]: 'source', stream: streams(s.utterances()), sampleRate: s.sampleRate };
  }
  if (isAsyncIterable(v)) return { [MARK]: 'stream', stream: streams(v as AsyncIterable<unknown>) };
  if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer || isDom(v, 'Blob') || isDom(v, 'ImageBitmap') || isDom(v, 'ImageData') || v instanceof Date) return v;
  if (Array.isArray(v)) return Promise.all(v.map((x) => toWire(x, streams, key)));
  const out: Record<string, unknown> = {};
  for (const [k, x] of Object.entries(v)) {
    const w = await toWire(x, streams, k);
    if (w !== undefined) out[k] = w;
  }
  return out;
}

/** Connect to a worker that called serveWorker(). */
export function connectWorker(worker: Worker | Port): EdgewiseWorker {
  const port = worker as Port;
  let seq = 0;
  let streamSeq = 0;
  const handlers = new Map<number, (m: ToMain) => void>();
  const pumps = new Map<number, () => void>();
  port.addEventListener('message', (e: MessageEvent) => {
    const m = e.data as ToMain;
    if (!m || typeof m !== 'object') return;
    if (m.type === 'stream-cancel') pumps.get(m.stream)?.();
    else if ('id' in m) handlers.get(m.id)?.(m);
  });
  (worker as { start?: () => void }).start?.();

  /** Pump a page-side iterable (a microphone, a text stream) into the worker until it ends or is stopped. */
  const openStream = (src: AsyncIterable<unknown>, owner?: Set<number>): number => {
    const stream = ++streamSeq;
    const it = src[Symbol.asyncIterator]();
    let stopped = false;
    const stop = () => {
      if (stopped) return;
      stopped = true;
      pumps.delete(stream);
      void it.return?.();
    };
    pumps.set(stream, stop);
    owner?.add(stream);
    (async () => {
      try {
        for (;;) {
          const r = await it.next();
          if (stopped) return;
          if (r.done) break;
          port.postMessage({ type: 'stream', stream, value: r.value } satisfies ToWorker);
        }
        port.postMessage({ type: 'stream', stream, done: true } satisfies ToWorker);
      } catch (err) {
        if (!stopped) port.postMessage({ type: 'stream', stream, done: true, error: err instanceof Error ? err.message : String(err) } satisfies ToWorker);
      } finally {
        pumps.delete(stream);
      }
    })();
    return stream;
  };

  interface CallOpts {
    signal?: AbortSignal;
    onChunk?: (c: unknown) => void;
    onEvent?: (e: RunEvent) => void;
    onProgress?: (e: LoadEvent) => void;
    tools?: Record<string, AnyTool>;
    approve?: (call: never) => boolean | Promise<boolean>;
    onBatch?: (p: { done: number; total: number }) => void;
  }

  async function call<R>(op: Op, rawArgs: unknown, o: CallOpts = {}): Promise<R> {
    const id = ++seq;
    if (o.signal?.aborted) throw new AbortError(undefined, { cause: o.signal.reason });
    const owned = new Set<number>();
    const stopStreams = () => {
      for (const s of owned) pumps.get(s)?.();
    };
    const args = await toWire(rawArgs, (src) => openStream(src, owned));
    const tools = o.tools
      ? Object.fromEntries(
          await Promise.all(
            Object.entries(o.tools).map(async ([name, t]) => [
              name,
              { description: t.description, inputSchema: await toJsonSchema(t.input), execute: !!t.execute },
            ]),
          ),
        )
      : undefined;
    return new Promise<R>((resolve, reject) => {
      const onAbort = () => {
        port.postMessage({ id, type: 'cancel' } satisfies ToWorker);
        handlers.delete(id);
        stopStreams();
        reject(new AbortError(undefined, { cause: o.signal?.reason }));
      };
      if (o.signal?.aborted) {
        stopStreams();
        reject(new AbortError(undefined, { cause: o.signal.reason }));
        return;
      }
      o.signal?.addEventListener('abort', onAbort, { once: true });
      handlers.set(id, async (m) => {
        if (m.type === 'chunk') o.onChunk?.(m.chunk);
        else if (m.type === 'batch') o.onBatch?.({ done: m.done, total: m.total });
        else if (m.type === 'event') {
          if (m.event.type === 'load') o.onProgress?.(m.event.event);
          o.onEvent?.(m.event);
        } else if (m.type === 'tool') {
          const t = o.tools?.[m.name];
          try {
            if (!t?.execute) throw new Error(`Tool "${m.name}" has no execute function.`);
            const v = validateWith(t.input, m.input);
            if (!v.success) throw new Error(v.error);
            const output = await t.execute(v.data, { signal: o.signal, call: { id: String(m.call), name: m.name } });
            port.postMessage({ id, type: 'tool-result', call: m.call, output: await toWire(output, openStream) } satisfies ToWorker);
          } catch (err) {
            port.postMessage({ id, type: 'tool-result', call: m.call, error: err instanceof Error ? err.message : String(err) } satisfies ToWorker);
          }
        } else if (m.type === 'approve') {
          let ok = false;
          try {
            ok = !!(await o.approve?.(m.request as never));
          } catch {
            ok = false;
          }
          port.postMessage({ id, type: 'approve-result', call: m.call, ok } satisfies ToWorker);
        } else {
          handlers.delete(id);
          o.signal?.removeEventListener('abort', onAbort);
          stopStreams();
          if (m.type === 'result') resolve(m.result as R);
          else reject(reviveError((m as Extract<ToMain, { type: 'error' }>).error));
        }
      });
      port.postMessage({ id, type: 'call', op, args, tools, approve: !!o.approve } satisfies ToWorker);
    });
  }

  const strip = <T extends object>(o: T, ...keys: string[]) => Object.fromEntries(Object.entries(o).filter(([k]) => !keys.includes(k)));
  const runCtx = <C>(ctx: RunContext<C>, options: { onProgress?: (e: LoadEvent) => void }) => ({
    signal: ctx.signal,
    onChunk: (c: unknown) => ctx.emit(c as C),
    onEvent: (e: RunEvent) => ctx.event(e),
    onProgress: options.onProgress,
  });

  return {
    generate<T>(options: GenerateOptions<T>) {
      return new Run<GenerateResult<T>, GenerateChunk<T>>(async (ctx) => {
        const schema = options.schema ? await toJsonSchema(options.schema) : undefined;
        const args = { ...strip(options, 'signal', 'onProgress', 'tools', 'approve', 'schema'), ...(schema ? { schema: { jsonSchema: schema } } : {}) };
        const r = await call<GenerateResult<T>>('generate', args, { ...runCtx(ctx, options), tools: options.tools, approve: options.approve as never });
        if (options.schema && r.object !== undefined) {
          const v = validateWith(options.schema, r.object);
          if (!v.success) throw new SchemaValidationError(`The output did not match the schema: ${v.error}`, { raw: r.text, issues: v.issues });
          r.object = v.data;
        }
        return r;
      }, options.signal);
    },
    evaluate(options) {
      return call('evaluate', strip(options, 'signal', 'onProgress'), { signal: options.signal, onProgress: options.onProgress });
    },
    embed(options: EmbedOneOptions | EmbedManyOptions) {
      return call(
        'embed',
        { ...strip(options, 'signal', 'onProgress', 'onBatch'), batches: 'onBatch' in options && !!options.onBatch },
        {
          signal: options.signal,
          onProgress: options.onProgress,
          onBatch: 'onBatch' in options ? options.onBatch : undefined,
        },
      ) as never;
    },
    cloneVoice(options: CloneVoiceOptions) {
      return call('cloneVoice', strip(options, 'signal', 'onProgress'), { signal: options.signal, onProgress: options.onProgress });
    },
    listVoices(model?: string) {
      return call('listVoices', { model });
    },
    speak(options: SpeakOptions) {
      return new SpeakRun(async (ctx: RunContext<SpeechChunk>) => {
        const r = await call<{ samples: Float32Array; sampleRate: number; info: SpeechAudio['info'] }>(
          'speak',
          strip(options, 'signal', 'onProgress'),
          runCtx(ctx, options),
        );
        return audioFromSamples(r.samples, r.sampleRate, r.info);
      }, options.signal);
    },
    paint(options: PaintOptions) {
      return new Run<PaintResult, PaintStep>(async (ctx) => {
        const r = await call<{ image: { data: Uint8ClampedArray; width: number; height: number }; seed: number; info: PaintResult['info'] }>(
          'paint',
          strip(options, 'signal', 'onProgress'),
          runCtx(ctx, options),
        );
        return { image: imageFromRgba(r.image.data, r.image.width, r.image.height), seed: r.seed, info: r.info };
      }, options.signal);
    },
    forecast(options: ForecastOptions) {
      return call('forecast', strip(options, 'signal', 'onProgress'), { signal: options.signal, onProgress: options.onProgress }) as never;
    },
    preload(models, o = {}) {
      return call('preload', { models, allowPreview: o.allowPreview }, { onProgress: o.onProgress });
    },
    unload(id) {
      return call('unload', { id });
    },
    capabilities() {
      return call('capabilities', {});
    },
    async configure(options) {
      const fn = hasFunction(options);
      if (fn) {
        throw new ConfigError(`"${fn}" is a function and cannot be sent to the worker.`, {
          hint: 'Set callbacks such as speak.onSynthesize by calling configure() inside the worker file.',
        });
      }
      return call('configure', options);
    },
    models() {
      return call('models', {});
    },
    terminate() {
      (worker as Worker).terminate?.();
      for (const [id, h] of handlers) h({ id, type: 'error', error: { name: 'AbortError', code: 'E_ABORT', message: 'The worker was terminated.' } });
      handlers.clear();
    },
  };
}

/* ---------------------------------------------------------------- worker side */

function serializeError(err: unknown): SerializedError {
  if (err instanceof SchemaValidationError)
    return { name: err.name, code: err.code, message: err.message, hint: err.hint, raw: err.raw, issues: plain(err.issues) };
  if (err instanceof EdgewiseError) return { name: err.name, code: err.code, message: err.message, hint: err.hint, retryable: err.retryable };
  if (err instanceof Error) return { name: err.name, message: err.message };
  return { name: 'Error', message: String(err) };
}

/** Structured-clone-safe copy of a value (functions and class instances with methods are reduced to data). */
function plain(v: unknown, depth = 0): unknown {
  if (v === null || typeof v !== 'object') return typeof v === 'function' ? undefined : v;
  if (depth > 12) return undefined;
  if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer || v instanceof Date || isDom(v, 'Blob') || isDom(v, 'ImageBitmap') || isDom(v, 'ImageData')) return v;
  if (Array.isArray(v)) return v.map((x) => plain(x, depth + 1));
  const out: Record<string, unknown> = {};
  for (const [k, x] of Object.entries(v)) {
    const p = plain(x, depth + 1);
    if (p !== undefined) out[k] = p;
  }
  return out;
}

/** Start serving Edgewise inside a worker. Define extra models with defineModel() before or after calling it. */
export function serveWorker(scope: Port = globalThis as unknown as Port): void {
  const running = new Map<number, AbortController>();
  const pending = new Map<string, (m: ToWorker) => void>();
  const streams = new Map<number, { push: (v: unknown) => void; end: (err?: string) => void }>();
  const early = new Map<number, ToWorker[]>();
  const post = (m: ToMain) => scope.postMessage(m);
  (scope as { start?: () => void }).start?.();

  function streamOf(stream: number): AsyncIterable<unknown> {
    const items: unknown[] = [];
    const waiters: ((r: IteratorResult<unknown>) => void)[] = [];
    let done = false;
    let error: string | undefined;
    const wake = () => {
      while (waiters.length && (items.length || done)) {
        const w = waiters.shift() as (r: IteratorResult<unknown>) => void;
        if (items.length) w({ value: items.shift(), done: false });
        else w({ value: undefined, done: true });
      }
    };
    streams.set(stream, {
      push: (v) => {
        items.push(v);
        wake();
      },
      end: (err) => {
        done = true;
        error = err;
        streams.delete(stream);
        wake();
      },
    });
    for (const m of early.get(stream) ?? []) handleStream(m as Extract<ToWorker, { type: 'stream' }>);
    early.delete(stream);
    return {
      [Symbol.asyncIterator]: () => ({
        next: async () => {
          if (items.length) return { value: items.shift(), done: false };
          if (done) {
            if (error) throw new Error(error);
            return { value: undefined, done: true };
          }
          const r = await new Promise<IteratorResult<unknown>>((res) => waiters.push(res));
          if (r.done && error) throw new Error(error);
          return r;
        },
      }),
    };
  }

  function handleStream(m: Extract<ToWorker, { type: 'stream' }>) {
    const s = streams.get(m.stream);
    if (!s) {
      const q = early.get(m.stream) ?? [];
      q.push(m);
      early.set(m.stream, q);
      return;
    }
    if (m.done) s.end(m.error);
    else s.push(m.value);
  }

  /** Streams revived for the call being prepared; cancelled on the page when the call ends. */
  let collecting: Set<number> | null = null;
  function revive(v: unknown): unknown {
    if (v === null || typeof v !== 'object') return v;
    const tag = (v as Record<string, unknown>)[MARK];
    if (tag === 'url') return new URL((v as { href: string }).href);
    if (tag === 'pcm') {
      const p = v as { samples: Float32Array; sampleRate: number };
      return { type: 'audio', audio: p.samples, sampleRate: p.sampleRate };
    }
    if (tag === 'source') {
      const s = v as { stream: number; sampleRate: number };
      collecting?.add(s.stream);
      const it = streamOf(s.stream) as AsyncIterable<Float32Array>;
      return { kind: 'audio-source', sampleRate: s.sampleRate, utterances: () => it } satisfies AudioSource;
    }
    if (tag === 'stream') {
      collecting?.add((v as { stream: number }).stream);
      return streamOf((v as { stream: number }).stream);
    }
    if (ArrayBuffer.isView(v) || v instanceof ArrayBuffer || v instanceof Date || isDom(v, 'Blob') || isDom(v, 'ImageBitmap') || isDom(v, 'ImageData'))
      return v;
    if (Array.isArray(v)) return v.map(revive);
    return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, revive(x)]));
  }

  // A round trip to the page. Resolves with a refusal if the run is cancelled meanwhile, so it never hangs.
  const rpc = (id: number, call: number, msg: ToMain, kind: 'tool-result' | 'approve-result', signal: AbortSignal) =>
    new Promise<ToWorker>((resolve) => {
      const key = `${id}:${kind}:${call}`;
      const cancelled = () => {
        pending.delete(key);
        resolve({ id, type: kind, call, error: 'The run was cancelled.', ok: false } as ToWorker);
      };
      if (signal.aborted) return cancelled();
      signal.addEventListener('abort', cancelled, { once: true });
      pending.set(key, (m) => {
        signal.removeEventListener('abort', cancelled);
        resolve(m);
      });
      post(msg);
    });

  async function run(m: Extract<ToWorker, { type: 'call' }>) {
    const ac = new AbortController();
    running.set(m.id, ac);
    const streamsOfCall = new Set<number>();
    collecting = streamsOfCall;
    const args = revive(m.args) as Record<string, unknown>;
    collecting = null;
    const onProgress = (e: LoadEvent) => post({ id: m.id, type: 'event', event: { type: 'load', event: e } });
    let callSeq = 0;
    const tools = m.tools
      ? Object.fromEntries(
          Object.entries(m.tools).map(([name, t]) => [
            name,
            {
              description: t.description,
              input: { jsonSchema: t.inputSchema as Record<string, unknown> },
              execute: t.execute
                ? async (input: unknown) => {
                    const call = ++callSeq;
                    const r = (await rpc(m.id, call, { id: m.id, type: 'tool', call, name, input: plain(input) }, 'tool-result', ac.signal)) as Extract<
                      ToWorker,
                      { type: 'tool-result' }
                    >;
                    if (r.error) throw new Error(r.error);
                    return revive(r.output);
                  }
                : undefined,
            } satisfies AnyTool,
          ]),
        )
      : undefined;
    const approve = m.approve
      ? async (request: unknown) => {
          const call = ++callSeq;
          const r = (await rpc(m.id, call, { id: m.id, type: 'approve', call, request: plain(request) }, 'approve-result', ac.signal)) as Extract<
            ToWorker,
            { type: 'approve-result' }
          >;
          return r.ok;
        }
      : undefined;
    const common = { ...args, signal: ac.signal, onProgress };
    const stream = async (r: Run<unknown, unknown>) => {
      const events = r.events;
      const pump = (async () => {
        // Load events already reach the page through onProgress; forwarding them here too would double them.
        for await (const e of events) if (e.type !== 'finish' && e.type !== 'load') post({ id: m.id, type: 'event', event: plain(e) as RunEvent });
      })().catch(() => {}); // the run's own rejection is reported below
      for await (const c of r) post({ id: m.id, type: 'chunk', chunk: plain(c) });
      await pump;
      return r;
    };
    try {
      let result: unknown;
      switch (m.op) {
        case 'generate': {
          const r = ew.generate({ ...(common as object), tools, approve } as never);
          result = await await stream(r as never);
          break;
        }
        case 'speak': {
          const r = ew.speak(common as never);
          const a = (await await stream(r as never)) as SpeechAudio;
          result = { samples: a.samples, sampleRate: a.sampleRate, info: a.info };
          break;
        }
        case 'paint': {
          const r = ew.paint(common as never);
          const p = (await await stream(r as never)) as PaintResult;
          result = { image: { data: p.image.data, width: p.image.width, height: p.image.height }, seed: p.seed, info: p.info };
          break;
        }
        case 'evaluate':
          result = await ew.evaluate(common as never);
          break;
        case 'embed': {
          const { batches, ...rest } = common as Record<string, unknown>;
          const onBatch = batches ? (p: { done: number; total: number }) => post({ id: m.id, type: 'batch', ...p }) : undefined;
          result = await ew.embed({ ...rest, onBatch } as never);
          break;
        }
        case 'cloneVoice':
          result = await ew.cloneVoice(common as never);
          break;
        case 'listVoices':
          result = await ew.listVoices((args as { model?: string }).model);
          break;
        case 'forecast':
          result = await ew.forecast(common as never);
          break;
        case 'preload': {
          const a = args as { models: string | string[]; allowPreview?: boolean };
          await ew.preload(Array.isArray(a.models) ? a.models : [a.models], { onProgress, allowPreview: a.allowPreview, signal: ac.signal });
          break;
        }
        case 'unload':
          await ew.unload((args as { id?: string }).id);
          break;
        case 'capabilities':
          result = await ew.capabilities();
          break;
        case 'configure':
          ew.configure(args as never);
          break;
        case 'models':
          result = registry.list().map((x: Manifest) => x.id);
          break;
        default:
          throw new ConfigError(`Unknown worker operation "${m.op as string}".`);
      }
      post({ id: m.id, type: 'result', result: plain(result) });
    } catch (err) {
      post({ id: m.id, type: 'error', error: serializeError(err) });
    } finally {
      running.delete(m.id);
      // Tell the page to stop any input stream this call did not read to the end (a microphone, say).
      for (const st of streamsOfCall) {
        if (streams.has(st) || early.has(st)) {
          scope.postMessage({ type: 'stream-cancel', stream: st } satisfies ToMain);
          streams.get(st)?.end();
          early.delete(st);
        }
      }
    }
  }

  scope.addEventListener('message', (e: MessageEvent) => {
    const m = e.data as ToWorker;
    if (!m || typeof m !== 'object') return;
    if (m.type === 'stream') handleStream(m);
    else if (m.type === 'call') void run(m);
    else if (m.type === 'cancel') running.get(m.id)?.abort(new AbortError('Run cancelled.'));
    else if (m.type === 'tool-result' || m.type === 'approve-result') {
      const key = `${m.id}:${m.type}:${m.call}`;
      pending.get(key)?.(m);
      pending.delete(key);
    }
  });
}
