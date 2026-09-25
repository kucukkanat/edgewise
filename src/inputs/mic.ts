import { createSileroStream, FRAME, RATE, type VadOptions, VadSegmenter } from '../backends/vad.ts';
import { PermissionError, UnsupportedDeviceError } from '../core/errors.ts';
import type { AudioSource } from '../core/parts.ts';
import { getPlatform } from '../core/runtime.ts';
import { resample } from '../platform/audio.ts';

export type { VadOptions } from '../backends/vad.ts';

export interface MicOptions {
  /** Output sample rate. Default 16000, which speech models expect. */
  sampleRate?: number;
  deviceId?: string;
  /** Split speech into utterances with Silero VAD. */
  vad?: boolean | VadOptions;
}

export interface Mic extends AudioSource {
  /** Start recording (push to talk). */
  start(): Promise<void>;
  /** Stop and return everything recorded since start(). */
  stop(): Promise<Float32Array>;
  /** Current input level, 0 to 1. */
  readonly level: number;
  readonly recording: boolean;
  /** Record a compressed copy with MediaRecorder. */
  toBlob(type?: string): Promise<Blob>;
  /** Release the microphone. */
  dispose(): void;
}

const WORKLET = `
class EdgewiseTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0];
    if (ch && ch[0]) {
      const n = ch[0].length, out = new Float32Array(n);
      for (let c = 0; c < ch.length; c++) { const d = ch[c]; for (let i = 0; i < n; i++) out[i] += d[i] / ch.length; }
      this.port.postMessage(out, [out.buffer]);
    }
    return true;
  }
}
registerProcessor('edgewise-tap', EdgewiseTap);
`;

class Queue<T> {
  private items: T[] = [];
  private waiters: ((v: IteratorResult<T>) => void)[] = [];
  private done = false;
  push(v: T) {
    const w = this.waiters.shift();
    if (w) w({ value: v, done: false });
    else this.items.push(v);
  }
  end() {
    this.done = true;
    for (const w of this.waiters.splice(0)) w({ value: undefined as T, done: true });
  }
  next(): Promise<IteratorResult<T>> {
    if (this.items.length) return Promise.resolve({ value: this.items.shift() as T, done: false });
    if (this.done) return Promise.resolve({ value: undefined as T, done: true });
    return new Promise((r) => this.waiters.push(r));
  }
}

/** Segment a stream of 16 kHz PCM chunks into utterances with Silero VAD. */
async function* segmentStream(chunks: AsyncIterable<Float32Array>, o: VadOptions): AsyncGenerator<Float32Array> {
  const vad = await createSileroStream(o.model);
  const seg = new VadSegmenter(o);
  let carry = new Float32Array(0);
  for await (const chunk of chunks) {
    const buf = new Float32Array(carry.length + chunk.length);
    buf.set(carry);
    buf.set(chunk, carry.length);
    let i = 0;
    for (; i + FRAME <= buf.length; i += FRAME) {
      const f = buf.slice(i, i + FRAME);
      const ev = seg.push(f, await vad.prob(f));
      if (ev?.type === 'end') yield ev.audio;
    }
    carry = buf.slice(i);
  }
  const last = seg.flush();
  if (last?.type === 'end') yield last.audio;
}

/**
 * Wrap any stream of PCM chunks as an audio source. Use this on Bun and Node,
 * for example with audio piped from ffmpeg.
 */
export function audioSource(chunks: AsyncIterable<Float32Array>, o: { sampleRate?: number; vad?: boolean | VadOptions } = {}): AudioSource {
  const rate = o.sampleRate ?? RATE;
  async function* at16k() {
    for await (const c of chunks) yield resample(c, rate, RATE);
  }
  return {
    kind: 'audio-source',
    sampleRate: RATE,
    utterances() {
      if (o.vad === false) return at16k();
      return segmentStream(at16k(), typeof o.vad === 'object' ? o.vad : {});
    },
  };
}

/** Open the microphone. Browser only; needs HTTPS or localhost. */
export async function mic(options: MicOptions = {}): Promise<Mic> {
  const p = getPlatform();
  const md = (globalThis as { navigator?: Navigator }).navigator?.mediaDevices;
  const Ctx = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext;
  if (!p.isBrowser || !md?.getUserMedia || !Ctx) {
    throw new UnsupportedDeviceError(`mic() needs a browser with microphone access; this is ${p.name}.`, {
      hint: 'On Bun and Node, pass audio as a Float32Array or WAV, or wrap a PCM stream with audioSource().',
    });
  }
  const outRate = options.sampleRate ?? RATE;
  let stream: MediaStream;
  try {
    stream = await md.getUserMedia({
      audio: { deviceId: options.deviceId, channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (err) {
    throw new PermissionError('Microphone access was denied or is unavailable.', {
      cause: err,
      hint: 'Allow microphone access for this site, then try again.',
    });
  }
  const ctx = new Ctx();
  const url = URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'edgewise-tap');
  src.connect(node);
  let level = 0;
  let recording = false;
  let recorded: Float32Array[] = [];
  const listeners = new Set<(chunk: Float32Array) => void>();
  node.port.onmessage = (e: MessageEvent<Float32Array>) => {
    const d = e.data;
    let peak = 0;
    for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]));
    level = level * 0.8 + peak * 0.2;
    if (recording) recorded.push(d);
    for (const l of listeners) l(d);
  };
  const vadOpts = options.vad === true ? {} : options.vad || null;

  const collect = () => {
    const n = recorded.reduce((a, b) => a + b.length, 0);
    const out = new Float32Array(n);
    let o = 0;
    for (const r of recorded) {
      out.set(r, o);
      o += r.length;
    }
    return resample(out, ctx.sampleRate, outRate);
  };

  const m: Mic = {
    kind: 'audio-source',
    sampleRate: outRate,
    get level() {
      return Math.min(1, level);
    },
    get recording() {
      return recording;
    },
    async start() {
      if (ctx.state === 'suspended') await ctx.resume();
      recorded = [];
      recording = true;
    },
    async stop() {
      recording = false;
      return collect();
    },
    async toBlob(type = 'audio/webm') {
      const rec = new MediaRecorder(stream, { mimeType: type });
      const parts: Blob[] = [];
      rec.ondataavailable = (e) => parts.push(e.data);
      const done = new Promise<Blob>((r) => {
        rec.onstop = () => r(new Blob(parts, { type }));
      });
      rec.start();
      await new Promise<void>((r) => {
        const check = () => (recording ? setTimeout(check, 100) : r());
        check();
      });
      rec.stop();
      return done;
    },
    utterances() {
      const q = new Queue<Float32Array>();
      const onChunk = (c: Float32Array) => q.push(resample(c, ctx.sampleRate, RATE));
      listeners.add(onChunk);
      if (ctx.state === 'suspended') void ctx.resume();
      const chunks: AsyncIterable<Float32Array> = { [Symbol.asyncIterator]: () => ({ next: () => q.next() }) };
      disposeHooks.push(() => {
        listeners.delete(onChunk);
        q.end();
      });
      if (!vadOpts) {
        // Without VAD, yield one clip per start()/stop() recording.
        async function* clips() {
          for (;;) {
            while (!recording) await new Promise((r) => setTimeout(r, 50));
            while (recording) await new Promise((r) => setTimeout(r, 50));
            yield collect();
          }
        }
        return clips();
      }
      return segmentStream(chunks, vadOpts);
    },
    dispose() {
      for (const h of disposeHooks.splice(0)) h();
      node.disconnect();
      src.disconnect();
      for (const t of stream.getTracks()) t.stop();
      void ctx.close();
    },
  };
  const disposeHooks: (() => void)[] = [];
  return m;
}
