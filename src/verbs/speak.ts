import { type ClonedVoice, chatterboxSentence, encodeVoice, loadSavedVoice, saveVoice } from '../backends/chatterbox.ts';
import { KOKORO_VOICES, kokoroSentence, type VoiceInfo, type VoiceMix, voiceVector } from '../backends/kokoro.ts';
import { splitSentences } from '../backends/kokoro-text.ts';
import { mockSpeak } from '../backends/mock.ts';
import { getConfig } from '../core/config.ts';
import { ConfigError, UnsupportedDeviceError, UnsupportedInputError } from '../core/errors.ts';
import type { AudioLike } from '../core/parts.ts';
import { registry, resolveManifest } from '../core/registry.ts';
import { Run, type RunContext } from '../core/run.ts';
import { getPlatform } from '../core/runtime.ts';
import type { CommonOptions, Manifest, ModelRef, RunInfo } from '../core/types.ts';
import { isAsyncIterable } from '../core/util.ts';
import { encodeWav } from '../platform/audio.ts';

export type Voice = VoiceInfo;
export type { ClonedVoice } from '../backends/chatterbox.ts';

/** Reference audio for a cloning model, with the speaker's consent. */
export interface VoiceReference {
  /** 3 to 10 seconds of clear speech from one speaker. */
  reference: AudioLike;
  /** Sample rate of `reference` when it is raw samples. */
  sampleRate?: number;
  /** You confirm the speaker agreed to have their voice cloned. Required. */
  consent: { attested: true; by?: string };
  /** Keep the voice for later as `voice: 'saved:<name>'`. */
  saveAs?: string;
}

/**
 * A preset voice ID, a weighted blend of presets, reference audio for a cloning model,
 * a voice from `cloneVoice()`, or `'saved:<name>'`.
 */
export type VoiceSpec = VoiceMix | VoiceReference | ClonedVoice;

export interface SpeakOptions extends CommonOptions {
  model: ModelRef;
  /** Text, or a stream of text deltas such as a running `generate()`. */
  input: string | AsyncIterable<string>;
  voice?: VoiceSpec;
  /** 0.5 to 2. Default 1. Kokoro only. */
  speed?: number;
  /** Cloning models: how expressive the speech is, 0 to 2. Default 0.5. */
  exaggeration?: number;
}

export interface SpeechAudio {
  samples: Float32Array;
  sampleRate: number;
  /** Seconds. */
  duration: number;
  /** 16-bit PCM WAV bytes. */
  toWav(): Uint8Array;
  toBlob(type?: 'audio/wav'): Blob;
  /** Browser only. Resolves when playback ends. */
  play(signal?: AbortSignal): Promise<void>;
  info: RunInfo;
}

export interface SpeechChunk {
  text: string;
  samples: Float32Array;
  sampleRate: number;
}

/** Wrap samples as SpeechAudio (used when results cross a worker boundary). */
export function audioFromSamples(samples: Float32Array, sampleRate: number, info: RunInfo): SpeechAudio {
  return makeAudio(samples, sampleRate, info);
}

function makeAudio(samples: Float32Array, sampleRate: number, info: RunInfo): SpeechAudio {
  return {
    samples,
    sampleRate,
    duration: samples.length / sampleRate,
    info,
    toWav: () => encodeWav(samples, sampleRate),
    toBlob: () => new Blob([encodeWav(samples, sampleRate) as Uint8Array<ArrayBuffer>], { type: 'audio/wav' }),
    play: (signal?: AbortSignal) => getPlatform().playAudio(samples, sampleRate, signal),
  };
}

/** A speak() run. `play()` starts playback with the first sentence, before synthesis finishes. */
export class SpeakRun extends Run<SpeechAudio, SpeechChunk> {
  /** Stream to the speakers (browser only). Resolves with the full audio when playback ends. */
  async play(): Promise<SpeechAudio> {
    const Ctx = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext;
    if (!getPlatform().isBrowser || !Ctx) {
      throw new UnsupportedDeviceError(`Audio playback is not available on ${getPlatform().name}.`, {
        hint: 'Await the run and write audio.toWav() to a file instead.',
      });
    }
    const ctx = new Ctx();
    if (ctx.state === 'suspended') await ctx.resume().catch(() => {});
    let at = ctx.currentTime + 0.05;
    const sources: AudioBufferSourceNode[] = [];
    const onAbort = () => {
      for (const s of sources) {
        try {
          s.stop();
        } catch {
          // already stopped
        }
      }
    };
    this.signal.addEventListener('abort', onAbort, { once: true });
    for await (const chunk of this) {
      if (!chunk.samples.length) continue;
      const buf = ctx.createBuffer(1, chunk.samples.length, chunk.sampleRate);
      buf.copyToChannel(chunk.samples as Float32Array<ArrayBuffer>, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      at = Math.max(at, ctx.currentTime + 0.01);
      src.start(at);
      at += buf.duration;
      sources.push(src);
    }
    const audio = await this;
    const wait = Math.max(0, at - ctx.currentTime);
    await new Promise((r) => setTimeout(r, wait * 1000 + 50));
    this.signal.removeEventListener('abort', onAbort);
    await ctx.close().catch(() => {});
    return audio;
  }
}

async function* sentencesOf(input: string | AsyncIterable<string>): AsyncGenerator<string> {
  if (typeof input === 'string') {
    for (const s of splitSentences(input, true).sentences) yield s;
    return;
  }
  let buf = '';
  for await (const delta of input) {
    if (typeof delta !== 'string') continue;
    buf += delta;
    const { sentences, rest } = splitSentences(buf);
    buf = rest;
    for (const s of sentences) yield s;
  }
  for (const s of splitSentences(buf, true).sentences) yield s;
}

/** Turn text, or a live stream of text, into speech. */
export function speak(options: SpeakOptions): SpeakRun {
  return new SpeakRun(async (ctx: RunContext<SpeechChunk>) => {
    if (typeof options.input !== 'string' && !isAsyncIterable(options.input)) {
      throw new ConfigError('speak() input must be a string or an async iterable of strings.');
    }
    const m = resolveManifest(options.model, { verb: 'speak', allowPreview: options.allowPreview, inputs: ['text'] });
    const speed = options.speed ?? 1;
    if (speed < 0.5 || speed > 2) throw new ConfigError('speed must be between 0.5 and 2.');
    const cloning = m.features?.includes('clone') ?? false;
    const spec: VoiceSpec | undefined = options.voice ?? (cloning ? undefined : ((m.config?.defaultVoice as string | undefined) ?? 'af_heart'));
    const common: CommonOptions = {
      signal: ctx.signal,
      device: options.device,
      dtype: options.dtype,
      allowPreview: options.allowPreview,
      onProgress: (e) => {
        options.onProgress?.(e);
        ctx.event({ type: 'load', event: e });
      },
    };
    const wantsClone = isReference(spec) || isCloned(spec) || (typeof spec === 'string' && spec.startsWith('saved:'));
    if (wantsClone && !cloning) {
      const cloners = registry
        .list({ verb: 'speak' })
        .filter((x) => x.features?.includes('clone'))
        .map((x) => x.id);
      throw new UnsupportedInputError(`"${m.id}" cannot clone voices.`, {
        hint: cloners.length ? `Use a cloning model: ${cloners.join(', ')}.` : 'No voice-cloning model is in the registry.',
      });
    }
    let say: (sentence: string) => Promise<{ samples: Float32Array; info: RunInfo }>;
    if (m.task === 'mock') say = async (sentence) => mockSpeak(m, sentence);
    else if (cloning) {
      if (!wantsClone) {
        throw new UnsupportedInputError(`"${m.id}" speaks in a cloned voice and needs reference audio.`, {
          hint: "Pass voice: { reference, consent: { attested: true } }, a voice from cloneVoice(), or 'saved:<name>'.",
        });
      }
      if (speed !== 1) throw new ConfigError(`"${m.id}" does not support speed.`);
      const voice = await resolveCloned(m, common, spec as VoiceReference | ClonedVoice | string);
      say = (sentence) => chatterboxSentence(m, common, sentence, voice, { exaggeration: options.exaggeration });
    } else {
      const voice = await voiceVector(m, spec as VoiceMix, ctx.signal);
      say = (sentence) => kokoroSentence(m, common, sentence, voice, speed);
    }
    const rate = (m.config?.sampleRate as number | undefined) ?? 24000;
    const pieces: Float32Array[] = [];
    let info: RunInfo = { model: m.id, device: 'cpu', dtype: 'unknown', backend: 'none' };
    for await (const sentence of sentencesOf(options.input)) {
      if (ctx.signal.aborted) break;
      const r = await say(sentence);
      info = r.info;
      pieces.push(r.samples);
      ctx.emit({ text: sentence, samples: r.samples, sampleRate: rate });
      ctx.event({ type: 'audio-chunk', text: sentence, samples: r.samples.length });
    }
    const n = pieces.reduce((a, b) => a + b.length, 0);
    const all = new Float32Array(n);
    let o = 0;
    for (const p of pieces) {
      all.set(p, o);
      o += p.length;
    }
    getConfig().speak.onSynthesize?.({ audio: all, sampleRate: rate, voice: spec });
    return makeAudio(all, rate, info);
  }, options.signal);
}

const isReference = (v: unknown): v is VoiceReference => typeof v === 'object' && v !== null && 'reference' in v;
const isCloned = (v: unknown): v is ClonedVoice => typeof v === 'object' && v !== null && (v as ClonedVoice).kind === 'cloned-voice';

function checkConsent(r: VoiceReference): void {
  if (r.consent?.attested !== true) {
    throw new ConfigError('Cloning a voice needs consent: pass consent: { attested: true } to confirm the speaker agreed.', {
      hint: 'Only clone voices of people who have given you permission.',
    });
  }
}

async function resolveCloned(m: Manifest, common: CommonOptions, spec: VoiceReference | ClonedVoice | string): Promise<ClonedVoice> {
  if (typeof spec === 'string') return loadSavedVoice(spec.slice('saved:'.length));
  if (isCloned(spec)) return spec;
  checkConsent(spec);
  const voice = await encodeVoice(m, common, spec.reference, spec.sampleRate);
  if (spec.saveAs) await saveVoice(spec.saveAs, voice);
  return voice;
}

export interface CloneVoiceOptions extends CommonOptions, VoiceReference {
  /** A cloning model. Default 'voice:clone'. */
  model?: ModelRef;
}

/**
 * Turn reference audio into a reusable voice. Pass the result as `voice` to speak(), or save it with `saveAs`.
 * Cloning needs the speaker's consent.
 */
export async function cloneVoice(options: CloneVoiceOptions): Promise<ClonedVoice> {
  checkConsent(options);
  const m = resolveManifest(options.model ?? 'voice:clone', { verb: 'speak', allowPreview: options.allowPreview, inputs: ['audio'] });
  if (!m.features?.includes('clone')) throw new UnsupportedInputError(`"${m.id}" cannot clone voices.`);
  const voice = await encodeVoice(m, options, options.reference, options.sampleRate);
  if (options.saveAs) await saveVoice(options.saveAs, voice);
  return voice;
}

/** Voices a speak model offers. */
export async function listVoices(model: ModelRef = 'voice:default'): Promise<Voice[]> {
  const m = resolveManifest(model, { verb: 'speak', allowPreview: true });
  if (m.task === 'kokoro') return KOKORO_VOICES.map((v) => ({ ...v }));
  return [];
}
