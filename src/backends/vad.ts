import { registry } from '../core/registry.ts';
import { getPlatform, loadCached, selectVariant } from '../core/runtime.ts';
import type { Manifest } from '../core/types.ts';
import type { OrtModule, OrtSession, OrtTensor } from '../platform/types.ts';
import { fetchModelFile } from './files.ts';

export interface VadOptions {
  /** Probability above which a frame counts as speech. Default 0.5. */
  positiveThreshold?: number;
  /** Probability below which a frame counts as silence. Default 0.35. */
  negativeThreshold?: number;
  /** Shortest utterance kept, in ms. Default 250. */
  minSpeechMs?: number;
  /** Silence that ends an utterance, in ms. Default 600. */
  redemptionMs?: number;
  /** Audio kept before speech starts, in ms. Default 300. */
  preSpeechPadMs?: number;
  /** VAD model. Default 'vad:default'. */
  model?: string;
}

export const FRAME = 512;
export const RATE = 16000;
const FRAME_MS = (FRAME / RATE) * 1000;

/**
 * Turns per-frame speech probabilities into utterances. Pure logic, no model:
 * the same state machine as @ricky0123/vad.
 */
export class VadSegmenter {
  private speaking = false;
  private redemption = 0;
  private speechFrames = 0;
  private current: Float32Array[] = [];
  private pre: Float32Array[] = [];
  private readonly pos: number;
  private readonly neg: number;
  private readonly minFrames: number;
  private readonly redemptionFrames: number;
  private readonly preFrames: number;

  constructor(o: VadOptions = {}) {
    this.pos = o.positiveThreshold ?? 0.5;
    this.neg = o.negativeThreshold ?? 0.35;
    this.minFrames = Math.max(1, Math.round((o.minSpeechMs ?? 250) / FRAME_MS));
    this.redemptionFrames = Math.max(1, Math.round((o.redemptionMs ?? 600) / FRAME_MS));
    this.preFrames = Math.max(0, Math.round((o.preSpeechPadMs ?? 300) / FRAME_MS));
  }

  /** Feed one frame and its probability. Returns an event when speech starts or an utterance ends. */
  push(frame: Float32Array, prob: number): { type: 'start' } | { type: 'end'; audio: Float32Array } | { type: 'misfire' } | null {
    if (!this.speaking) {
      this.pre.push(frame);
      if (this.pre.length > this.preFrames + 1) this.pre.shift();
      if (prob >= this.pos) {
        this.speaking = true;
        this.redemption = 0;
        this.speechFrames = 1;
        this.current = [...this.pre];
        this.pre = [];
        return { type: 'start' };
      }
      return null;
    }
    this.current.push(frame);
    if (prob >= this.pos) {
      this.speechFrames++;
      this.redemption = 0;
    } else if (prob < this.neg) {
      this.redemption++;
      if (this.redemption >= this.redemptionFrames) return this.end();
    }
    return null;
  }

  /** Close any open utterance (end of stream). */
  flush(): { type: 'end'; audio: Float32Array } | { type: 'misfire' } | null {
    return this.speaking ? this.end() : null;
  }

  private end(): { type: 'end'; audio: Float32Array } | { type: 'misfire' } {
    this.speaking = false;
    this.redemption = 0;
    const frames = this.current;
    this.current = [];
    if (this.speechFrames < this.minFrames) return { type: 'misfire' };
    const n = frames.reduce((a, b) => a + b.length, 0);
    const audio = new Float32Array(n);
    let o = 0;
    for (const f of frames) {
      audio.set(f, o);
      o += f.length;
    }
    return { type: 'end', audio };
  }
}

interface Silero {
  ort: OrtModule;
  session: OrtSession;
  stateName: string;
  stateShape: number[];
}

export async function loadSilero(m: Manifest, signal?: AbortSignal): Promise<Silero> {
  const sel = await selectVariant(m, { device: getPlatform().isBrowser ? 'wasm' : 'cpu' });
  return loadCached(`${m.id}|cpu`, { manifest: m, selection: sel, progress: () => {}, signal }, async () => {
    const bytes = await fetchModelFile(m, 'onnx/model.onnx', { signal });
    const ort = await getPlatform().loadOrt();
    const session = await ort.InferenceSession.create(bytes, getPlatform().isBrowser ? { executionProviders: ['wasm'] } : {});
    const stateName = session.inputNames.find((n) => n === 'state' || n === 'h') ?? 'state';
    return { ort, session, stateName, stateShape: [2, 1, 128] };
  });
}

/** A streaming Silero VAD. Call `prob(frame)` with 512-sample frames at 16 kHz. */
export async function createSileroStream(
  modelId = 'vad:default',
  signal?: AbortSignal,
): Promise<{ prob(frame: Float32Array): Promise<number>; reset(): void }> {
  const m = registry.get(modelId);
  const s = await loadSilero(m, signal);
  const zeros = () => new s.ort.Tensor('float32', new Float32Array(2 * 128), s.stateShape);
  let state: OrtTensor = zeros();
  const sr = new s.ort.Tensor('int64', BigInt64Array.from([BigInt(RATE)]), []);
  return {
    async prob(frame: Float32Array) {
      const input = new s.ort.Tensor('float32', frame, [1, frame.length]);
      const feeds: Record<string, OrtTensor> = { input, sr, [s.stateName]: state };
      const out = await s.session.run(feeds);
      const outName = s.session.outputNames.find((n) => n !== 'stateN' && n !== 'hn') ?? s.session.outputNames[0];
      state = out.stateN ?? out.hn ?? state;
      return Number(out[outName].data[0]);
    },
    reset() {
      state = zeros();
    },
  };
}

/** Split recorded audio into speech segments (start and end in seconds). */
export async function detectSpeech(samples: Float32Array, o: VadOptions = {}): Promise<{ start: number; end: number; audio: Float32Array }[]> {
  const vad = await createSileroStream(o.model);
  const seg = new VadSegmenter(o);
  const out: { start: number; end: number; audio: Float32Array }[] = [];
  let startFrame = 0;
  const frames = Math.floor(samples.length / FRAME);
  for (let i = 0; i < frames; i++) {
    const f = samples.subarray(i * FRAME, (i + 1) * FRAME);
    const p = await vad.prob(f.slice());
    const ev = seg.push(f, p);
    if (ev?.type === 'start') startFrame = i;
    if (ev?.type === 'end') out.push({ start: (startFrame * FRAME) / RATE - (o.preSpeechPadMs ?? 300) / 1000, end: ((i + 1) * FRAME) / RATE, audio: ev.audio });
  }
  const last = seg.flush();
  if (last?.type === 'end') out.push({ start: (startFrame * FRAME) / RATE, end: samples.length / RATE, audio: last.audio });
  return out.map((s) => ({ ...s, start: Math.max(0, s.start) }));
}
