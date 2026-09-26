import { ConfigError, throwIfAborted, UnsupportedInputError } from '../core/errors.ts';
import type { AudioLike } from '../core/parts.ts';
import { getPlatform } from '../core/runtime.ts';
import type { CommonOptions, Manifest, RunInfo } from '../core/types.ts';
import { serialize } from '../core/util.ts';
import { toAudio } from './media.ts';
import { getTransformers, loadTjs, repoOf } from './transformers.ts';

/**
 * Chatterbox (Resemble AI): zero-shot voice cloning. The speech encoder turns a few seconds of reference audio
 * into speaker conditioning; the language model writes speech tokens; the conditional decoder renders 24 kHz audio.
 */

const RATE = 24000;

type TensorLike = { data: ArrayLike<number | bigint>; dims: number[]; type: string };

interface Bundle {
  model: {
    encode_speech(audio_values: unknown): Promise<Record<'audio_features' | 'audio_tokens' | 'speaker_embeddings' | 'speaker_features', TensorLike>>;
    generate(params: Record<string, unknown>): Promise<TensorLike>;
    dispose?: () => Promise<unknown>;
  };
  tokenizer: (text: string) => Record<string, unknown>;
}

/** The speaker conditioning Chatterbox computes from reference audio. Reusable across sentences and sessions. */
export interface ClonedVoice {
  kind: 'cloned-voice';
  model: string;
  audio_features: SerializedTensor;
  audio_tokens: SerializedTensor;
  speaker_embeddings: SerializedTensor;
  speaker_features: SerializedTensor;
}

interface SerializedTensor {
  type: 'float32' | 'int64';
  dims: number[];
  data: Float32Array | BigInt64Array;
}

export async function loadChatterbox(m: Manifest, opts: CommonOptions) {
  return loadTjs<Bundle>(
    m,
    opts,
    async (t, base) => {
      const { repo } = repoOf(m);
      const cls = (t as unknown as { ChatterboxModel: { from_pretrained(repo: string, o: unknown): Promise<unknown> } }).ChatterboxModel;
      const [model, tokenizer] = await Promise.all([
        cls.from_pretrained(repo, base),
        t.AutoTokenizer.from_pretrained(repo, { revision: base.revision, progress_callback: base.progress_callback as never }),
      ]);
      return { model: model as Bundle['model'], tokenizer: tokenizer as unknown as Bundle['tokenizer'] };
    },
    async (b) => {
      await b.model.dispose?.();
    },
  );
}

/** Reference audio → speaker conditioning. Trims to the model's limit and refuses clips that are too short. */
export async function encodeVoice(m: Manifest, opts: CommonOptions, reference: AudioLike, sampleRate?: number): Promise<ClonedVoice> {
  const cfg = (m.config ?? {}) as { minReferenceSeconds?: number; maxReferenceSeconds?: number };
  let samples = await toAudio(reference, sampleRate, RATE);
  const min = cfg.minReferenceSeconds ?? 3;
  const max = cfg.maxReferenceSeconds ?? 10;
  if (samples.length < min * RATE) {
    throw new UnsupportedInputError(`The reference audio is ${(samples.length / RATE).toFixed(1)} s long; "${m.id}" needs at least ${min} s.`, {
      hint: `Record ${min} to ${max} seconds of clear speech from one speaker.`,
    });
  }
  if (samples.length > max * RATE) samples = samples.subarray(0, max * RATE);
  const t = await getTransformers();
  const loaded = await loadChatterbox(m, opts);
  const out = await serialize(`clone:${loaded.info.model}:${loaded.info.device}`, () =>
    loaded.value.model.encode_speech(new t.Tensor('float32', samples, [1, samples.length])),
  );
  const ser = (x: TensorLike): SerializedTensor =>
    x.type === 'int64'
      ? { type: 'int64', dims: [...x.dims], data: BigInt64Array.from(x.data as ArrayLike<bigint>) }
      : { type: 'float32', dims: [...x.dims], data: Float32Array.from(x.data as ArrayLike<number>) };
  return {
    kind: 'cloned-voice',
    model: m.id,
    audio_features: ser(out.audio_features),
    audio_tokens: ser(out.audio_tokens),
    speaker_embeddings: ser(out.speaker_embeddings),
    speaker_features: ser(out.speaker_features),
  };
}

/** Speak one sentence in a cloned voice. Returns 24 kHz mono samples. */
export async function chatterboxSentence(
  m: Manifest,
  opts: CommonOptions,
  text: string,
  voice: ClonedVoice,
  o: { exaggeration?: number; maxTokens?: number } = {},
): Promise<{ samples: Float32Array; info: RunInfo }> {
  if (voice.model !== m.id) throw new ConfigError(`This voice was made with "${voice.model}" and cannot be used with "${m.id}".`);
  const t = await getTransformers();
  const loaded = await loadChatterbox(m, opts);
  const tensor = (s: SerializedTensor) => new t.Tensor(s.type, s.data, s.dims);
  const ids = loaded.value.tokenizer(text);
  // Roughly 25 speech tokens per second; allow generous room for slow speech.
  const words = text.split(/\s+/).filter(Boolean).length;
  const maxTokens = o.maxTokens ?? Math.min(1000, 60 + words * 25);
  throwIfAborted(opts.signal);
  const stopper = new (t as unknown as { InterruptableStoppingCriteria: new () => { interrupt(): void } }).InterruptableStoppingCriteria();
  const onAbort = () => stopper.interrupt();
  opts.signal?.addEventListener('abort', onAbort, { once: true });
  const wav = await serialize(`clone:${loaded.info.model}:${loaded.info.device}`, () =>
    loaded.value.model.generate({
      ...ids,
      audio_features: tensor(voice.audio_features),
      audio_tokens: tensor(voice.audio_tokens),
      speaker_embeddings: tensor(voice.speaker_embeddings),
      speaker_features: tensor(voice.speaker_features),
      exaggeration: o.exaggeration ?? 0.5,
      max_new_tokens: maxTokens,
      repetition_penalty: 1.2,
      do_sample: false,
      stopping_criteria: [stopper],
    }),
  ).finally(() => opts.signal?.removeEventListener('abort', onAbort));
  throwIfAborted(opts.signal);
  return { samples: Float32Array.from(wav.data as ArrayLike<number>), info: loaded.info };
}

/* ---------- saved voices ---------- */

const MAGIC = 'EWVOICE1';

export function voiceToBytes(v: ClonedVoice): Uint8Array {
  const keys = ['audio_features', 'audio_tokens', 'speaker_embeddings', 'speaker_features'] as const;
  const header = { model: v.model, tensors: keys.map((k) => ({ name: k, type: v[k].type, dims: v[k].dims, bytes: v[k].data.byteLength })) };
  const h = new TextEncoder().encode(JSON.stringify(header));
  const total = 8 + 4 + h.length + keys.reduce((a, k) => a + v[k].data.byteLength, 0);
  const out = new Uint8Array(total);
  out.set(new TextEncoder().encode(MAGIC), 0);
  new DataView(out.buffer).setUint32(8, h.length, true);
  out.set(h, 12);
  let o = 12 + h.length;
  for (const k of keys) {
    const d = v[k].data;
    out.set(new Uint8Array(d.buffer, d.byteOffset, d.byteLength), o);
    o += d.byteLength;
  }
  return out;
}

const TENSORS = ['audio_features', 'audio_tokens', 'speaker_embeddings', 'speaker_features'] as const;

export function voiceFromBytes(b: Uint8Array): ClonedVoice {
  const bad = (why: string) => new ConfigError(`This saved voice is damaged (${why}).`, { hint: 'Clone the voice again with saveAs.' });
  if (b.length < 12 || new TextDecoder().decode(b.subarray(0, 8)) !== MAGIC) throw new ConfigError('Not a saved Edgewise voice.');
  const len = new DataView(b.buffer, b.byteOffset).getUint32(8, true);
  if (12 + len > b.length) throw bad('truncated header');
  let header: {
    model: string;
    tensors: { name: (typeof TENSORS)[number]; type: 'float32' | 'int64'; dims: number[]; bytes: number }[];
  };
  try {
    header = JSON.parse(new TextDecoder().decode(b.subarray(12, 12 + len)));
  } catch {
    throw bad('unreadable header');
  }
  if (typeof header?.model !== 'string' || !Array.isArray(header.tensors)) throw bad('unexpected header');
  let o = 12 + len;
  const v = { kind: 'cloned-voice', model: header.model } as ClonedVoice;
  for (const t of header.tensors) {
    if (!TENSORS.includes(t.name) || (t.type !== 'float32' && t.type !== 'int64')) throw bad(`unexpected tensor ${String(t.name)}`);
    const size = t.dims.reduce((a, d) => a * d, 1) * (t.type === 'int64' ? 8 : 4);
    if (size !== t.bytes || o + t.bytes > b.length) throw bad('truncated data');
    const raw = b.slice(o, o + t.bytes).buffer;
    v[t.name] = { type: t.type, dims: t.dims, data: t.type === 'int64' ? new BigInt64Array(raw) : new Float32Array(raw) };
    o += t.bytes;
  }
  for (const k of TENSORS) if (!v[k]) throw bad(`missing ${k}`);
  return v;
}

const fileOf = (name: string) => `edgewise-voice-${name.replace(/[^\w.-]/g, '_')}.bin`;

export async function saveVoice(name: string, v: ClonedVoice): Promise<void> {
  await getPlatform().store(fileOf(name), voiceToBytes(v));
}

export async function loadSavedVoice(name: string): Promise<ClonedVoice> {
  const b = await getPlatform().store(fileOf(name));
  if (!b) throw new ConfigError(`No saved voice named "${name}".`, { hint: 'Clone it first with voice: { reference, consent, saveAs }.' });
  return voiceFromBytes(b);
}
