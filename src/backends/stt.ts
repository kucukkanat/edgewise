import { AbortError } from '../core/errors.ts';
import type { CommonOptions, Manifest, RunInfo } from '../core/types.ts';
import { serialize } from '../core/util.ts';
import { getTransformers, loadTjs, repoOf } from './transformers.ts';

export interface Segment {
  start: number;
  end: number;
  text: string;
}

export interface SttOptions {
  timestamps?: false | 'segment' | 'word';
  language?: string;
  signal?: AbortSignal;
  onText?: (delta: string) => void;
  /** Long audio: called as each window finishes, with its segments (absolute times) and progress. */
  onSegment?: (segments: Segment[], progress: { done: number; total: number }) => void;
}

const RATE = 16000;

/** A slice of the input, in samples. */
export interface Window {
  start: number;
  end: number;
}

export interface SplitOptions {
  /** Longest window. Default 28 s: Whisper's window is 30 s and Moonshine degrades past it. */
  maxSeconds?: number;
  /** Shortest window before a cut is allowed. Default 12 s. */
  minSeconds?: number;
  /** RMS below which a 50 ms frame counts as silence. Default 0.003 (about -50 dBFS). */
  silence?: number;
  /** Audio kept around speech when a window's silent edges are trimmed. Default 0.3 s. */
  padSeconds?: number;
}

/**
 * Splits long 16 kHz audio at its quietest 50 ms between `minSeconds` and `maxSeconds`, so cuts land in pauses
 * rather than mid-word, then trims each window's silent edges and drops windows with no sound. Moonshine tiny
 * returns nothing for a window that opens with seconds of silence, and trimmed windows also give timestamps
 * that start at the speech. It is plain RMS with no model, so an hour of audio splits in milliseconds.
 */
export function splitOnSilence(samples: Float32Array, o: SplitOptions = {}): Window[] {
  const max = Math.round((o.maxSeconds ?? 28) * RATE);
  const min = Math.min(max, Math.round((o.minSeconds ?? 12) * RATE));
  const frame = RATE / 20;
  const rms = (a: number, b: number) => {
    let sum = 0;
    for (let i = a; i < b; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / Math.max(1, b - a));
  };
  const windows: Window[] = [];
  for (let start = 0; start < samples.length; ) {
    if (samples.length - start <= max) {
      windows.push({ start, end: samples.length });
      break;
    }
    let cut = start + max;
    let quietest = Number.POSITIVE_INFINITY;
    for (let f = start + min; f + frame <= start + max; f += frame) {
      const e = rms(f, f + frame);
      // `<=` keeps the latest of equally quiet frames: longer windows mean fewer model passes.
      if (e <= quietest) {
        quietest = e;
        cut = f + frame / 2;
      }
    }
    windows.push({ start, end: cut });
    start = cut;
  }
  const silence = o.silence ?? 0.003;
  const pad = Math.round((o.padSeconds ?? 0.3) * RATE);
  const loud = (f: number) => rms(f, Math.min(f + frame, samples.length)) >= silence;
  return windows.flatMap((w) => {
    let a = w.start;
    while (a < w.end && !loud(a)) a += frame;
    if (a >= w.end) return [];
    let b = w.end;
    while (b - frame > a && !loud(b - frame)) b -= frame;
    return [{ start: Math.max(w.start, a - pad), end: Math.min(w.end, b + pad) }];
  });
}

type Asr = ((audio: Float32Array, o: Record<string, unknown>) => Promise<{ text: string; chunks?: { timestamp: [number, number | null]; text: string }[] }>) & {
  tokenizer: unknown;
  dispose?: () => Promise<void>;
};

export async function loadAsr(m: Manifest, opts: CommonOptions) {
  return loadTjs<Asr>(
    m,
    opts,
    async (tj, base) => {
      const { repo } = repoOf(m);
      return (await tj.pipeline('automatic-speech-recognition', repo, base as never)) as unknown as Asr;
    },
    async (p) => {
      await p.dispose?.();
    },
  );
}

export async function transcribeSamples(
  m: Manifest,
  opts: CommonOptions,
  samples: Float32Array,
  o: SttOptions,
): Promise<{ text: string; segments?: Segment[]; language?: string; info: RunInfo; outputTokens: number }> {
  const t = await getTransformers();
  const loaded = await loadAsr(m, opts);
  const asr = loaded.value;
  const whisper = !!m.config?.whisper;
  const language = whisper ? (o.language && o.language !== 'auto' ? o.language : undefined) : (m.config?.language as string | undefined);
  // Under 0.1 s there is nothing to transcribe, and the encoders' first convolution rejects such short input.
  if (samples.length < 1600) return { text: '', segments: [], language, info: loaded.info, outputTokens: 0 };
  let tokens = 0;

  // One model pass over at most ~30 s, with segment times relative to `offset` seconds.
  const pass = async (audio: Float32Array, offset: number): Promise<{ text: string; segments: Segment[] }> => {
    const call: Record<string, unknown> = {};
    if (whisper) {
      if (o.timestamps) call.return_timestamps = o.timestamps === 'word' ? 'word' : true;
      if (m.config?.multilingual) {
        if (o.language && o.language !== 'auto') call.language = o.language;
        call.task = 'transcribe';
      }
    }
    if (o.onText) {
      call.streamer = new t.TextStreamer(
        asr.tokenizer as never,
        {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (s: string) => o.onText?.(s),
          token_callback_function: () => {
            tokens++;
          },
        } as never,
      );
    }
    const out = await serialize(`stt:${loaded.info.model}:${loaded.info.device}`, () => asr(audio, call));
    const text = out.text.trim();
    const end = offset + audio.length / RATE;
    const segments = out.chunks?.length
      ? out.chunks.map((c) => ({ start: offset + c.timestamp[0], end: c.timestamp[1] === null ? end : offset + c.timestamp[1], text: c.text.trim() }))
      : [{ start: offset, end, text }];
    return { text, segments };
  };

  // Long audio is cut at pauses into windows and transcribed window by window, for every model: Moonshine
  // degrades past ~30 s in one pass, and per-window passes give progress, early text, abort between windows
  // and skip silence, which Whisper's own strided chunking does not.
  const windows = samples.length > 30 * RATE ? splitOnSilence(samples) : [{ start: 0, end: samples.length }];
  const texts: string[] = [];
  const segments: Segment[] = [];
  for (const [i, w] of windows.entries()) {
    if (o.signal?.aborted) throw new AbortError(undefined, { cause: o.signal.reason });
    if (i > 0 && texts.length) o.onText?.(' ');
    const r = await pass(samples.subarray(w.start, w.end), w.start / RATE);
    const kept = r.segments.filter((x) => x.text);
    if (r.text) texts.push(r.text);
    segments.push(...kept);
    o.onSegment?.(kept, { done: i + 1, total: windows.length });
  }
  const text = texts.join(' ');
  return { text, segments: o.timestamps ? segments : undefined, language, info: loaded.info, outputTokens: tokens || Math.ceil(text.length / 4) };
}
