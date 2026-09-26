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
  // Under 0.1 s there is nothing to transcribe, and the encoders' first convolution rejects such short input.
  if (samples.length < 1600) return { text: '', segments: [], info: loaded.info, outputTokens: 0 };
  const whisper = !!m.config?.whisper;
  const call: Record<string, unknown> = {};
  if (whisper) {
    call.chunk_length_s = samples.length > 30 * 16000 ? 30 : undefined;
    call.stride_length_s = samples.length > 30 * 16000 ? 5 : undefined;
    if (o.timestamps) call.return_timestamps = o.timestamps === 'word' ? 'word' : true;
    if (m.config?.multilingual) {
      if (o.language && o.language !== 'auto') call.language = o.language;
      call.task = 'transcribe';
    }
  }
  let tokens = 0;
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
  for (const k of Object.keys(call)) if (call[k] === undefined) delete call[k];
  const out = await serialize(`stt:${loaded.info.model}:${loaded.info.device}`, () => asr(samples, call));
  const text = out.text.trim();
  let segments: Segment[] | undefined;
  if (o.timestamps) {
    if (out.chunks?.length) {
      segments = out.chunks.map((c) => ({ start: c.timestamp[0], end: c.timestamp[1] ?? samples.length / 16000, text: c.text.trim() }));
    } else {
      segments = [{ start: 0, end: samples.length / 16000, text }];
    }
  }
  return {
    text,
    segments,
    language: whisper ? (o.language && o.language !== 'auto' ? o.language : undefined) : (m.config?.language as string | undefined),
    info: loaded.info,
    outputTokens: tokens || Math.ceil(text.length / 4),
  };
}
