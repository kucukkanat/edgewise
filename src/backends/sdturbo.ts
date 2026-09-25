import { AbortError, toEdgewiseError } from '../core/errors.ts';
import { dtypeLabel, getPlatform, selectVariant } from '../core/runtime.ts';
import type { CommonOptions, Device, LoadEvent, Manifest, RunInfo } from '../core/types.ts';
import { serialize } from '../core/util.ts';
import type { OrtSession } from '../platform/types.ts';
import { fetchModelFile } from './files.ts';
import { getTransformers } from './transformers.ts';

const TOKENIZER = { repo: 'Xenova/clip-vit-base-patch16', revision: '342fdf2f67aded64d138ff074745fb4a5d2bba5f' };

/** Seeded normal random numbers (mulberry32 + Box–Muller). */
export function gaussian(seed: number): () => number {
  let a = seed >>> 0 || 1;
  const uni = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    const u = uni() || 1e-12;
    const v = uni();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

/** Sigma for a timestep under SD's scaled-linear beta schedule. */
export function sigmaAt(t: number): number {
  const b0 = Math.sqrt(0.00085);
  const b1 = Math.sqrt(0.012);
  let prod = 1;
  for (let i = 0; i <= t; i++) {
    const beta = (b0 + ((b1 - b0) * i) / 999) ** 2;
    prod *= 1 - beta;
  }
  return Math.sqrt((1 - prod) / prod);
}

/** Cheap RGB preview straight from latents (no VAE). */
export function latentPreview(lat: Float32Array, L: number): { data: Uint8ClampedArray; width: number; height: number } {
  const F = [
    [0.298, 0.207, 0.208],
    [0.187, 0.286, 0.173],
    [-0.158, 0.189, 0.264],
    [-0.184, -0.271, -0.473],
  ];
  const data = new Uint8ClampedArray(L * L * 4);
  for (let i = 0; i < L * L; i++) {
    for (let c = 0; c < 3; c++) {
      let v = 0;
      for (let k = 0; k < 4; k++) v += lat[k * L * L + i] * F[k][c];
      data[i * 4 + c] = Math.round((v * 0.5 + 0.5) * 255);
    }
    data[i * 4 + 3] = 255;
  }
  return { data, width: L, height: L };
}

const sessions = new Map<string, Promise<OrtSession>>();

async function session(m: Manifest, part: 'text_encoder' | 'unet' | 'vae_decoder', device: Device, onProgress?: (e: LoadEvent) => void, signal?: AbortSignal) {
  const key = `${m.id}|${part}|${device}`;
  let s = sessions.get(key);
  if (!s) {
    s = (async () => {
      const bytes = await fetchModelFile(m, `${part}/model.onnx`, { signal, onProgress });
      onProgress?.({ type: 'compile', model: m.id });
      const ort = await getPlatform().loadOrt();
      const eps = device === 'webgpu' ? ['webgpu'] : getPlatform().isBrowser ? ['wasm'] : ['cpu'];
      // On CPU, keep peak memory down: full optimization folds fp16-to-fp32 casts into a second copy of the weights.
      const cpu = eps[0] !== 'webgpu';
      return ort.InferenceSession.create(bytes, {
        executionProviders: eps,
        graphOptimizationLevel: cpu ? 'basic' : 'all',
        ...(cpu ? { enableCpuMemArena: false } : {}),
      });
    })();
    sessions.set(key, s);
    s.catch(() => sessions.delete(key));
  }
  return s;
}

async function releaseSession(m: Manifest, part: string, device: Device) {
  const key = `${m.id}|${part}|${device}`;
  const s = sessions.get(key);
  sessions.delete(key);
  try {
    await (await s)?.release?.();
  } catch {
    // ignore
  }
}

export interface PaintParams {
  prompt: string;
  width: number;
  height: number;
  steps: number;
  seed: number;
  signal?: AbortSignal;
  onStep?: (step: number, total: number, latents: Float32Array, L: [number, number]) => void;
}

export async function sdTurbo(
  m: Manifest,
  opts: CommonOptions,
  p: PaintParams,
): Promise<{ rgb: Uint8ClampedArray; width: number; height: number; info: RunInfo }> {
  const sel = await selectVariant(m, opts);
  const device = sel.device;
  const lowMemory = device !== 'webgpu';
  const t = await getTransformers();
  const tok = await t.AutoTokenizer.from_pretrained(TOKENIZER.repo, { revision: TOKENIZER.revision });
  const info: RunInfo = {
    model: m.id,
    device,
    dtype: dtypeLabel(sel.dtype),
    backend: getPlatform().isBrowser ? `onnxruntime-web:${device}` : `onnxruntime-node:${device}`,
  };
  const ort = await getPlatform().loadOrt();
  const check = () => {
    if (p.signal?.aborted) throw new AbortError(undefined, { cause: p.signal.reason });
  };
  return serialize(`paint:${m.id}:${device}`, async () => {
    try {
      const enc = tok(p.prompt, { padding: 'max_length', max_length: 77, truncation: true }) as unknown as { input_ids: { data: ArrayLike<bigint | number> } };
      const ids = Int32Array.from(Array.from(enc.input_ids.data, (x) => Number(x)));
      const te = await session(m, 'text_encoder', device, opts.onProgress, p.signal);
      const hidden = (await te.run({ input_ids: new ort.Tensor('int32', ids, [1, 77]) })).last_hidden_state;
      if (lowMemory) await releaseSession(m, 'text_encoder', device);
      check();
      const Lh = Math.round(p.height / 8);
      const Lw = Math.round(p.width / 8);
      const n = 4 * Lh * Lw;
      const rnd = gaussian(p.seed);
      const steps = Math.max(1, Math.min(4, p.steps));
      const timesteps = Array.from({ length: steps }, (_, i) => Math.round(999 - (i * 1000) / steps));
      const sigmas = timesteps.map(sigmaAt);
      let x = new Float32Array(n);
      for (let i = 0; i < n; i++) x[i] = rnd() * sigmas[0];
      const unet = await session(m, 'unet', device, opts.onProgress, p.signal);
      let x0 = new Float32Array(n);
      for (let s = 0; s < steps; s++) {
        check();
        const sigma = sigmas[s];
        const scale = 1 / Math.sqrt(sigma * sigma + 1);
        const inp = new Float32Array(n);
        for (let i = 0; i < n; i++) inp[i] = x[i] * scale;
        const out = await unet.run({
          sample: new ort.Tensor('float32', inp, [1, 4, Lh, Lw]),
          timestep: new ort.Tensor('int64', BigInt64Array.from([BigInt(timesteps[s])]), [1]),
          encoder_hidden_states: hidden,
        });
        const eps = out.out_sample.data as Float32Array;
        x0 = new Float32Array(n);
        for (let i = 0; i < n; i++) x0[i] = x[i] - sigma * eps[i];
        p.onStep?.(s + 1, steps, x0, [Lh, Lw]);
        if (s < steps - 1) {
          const next = sigmas[s + 1];
          x = new Float32Array(n);
          for (let i = 0; i < n; i++) x[i] = x0[i] + next * rnd();
        }
      }
      if (lowMemory) await releaseSession(m, 'unet', device);
      check();
      const scaleVae = (m.config?.vaeScale as number | undefined) ?? 0.18215;
      const z = new Float32Array(n);
      for (let i = 0; i < n; i++) z[i] = x0[i] / scaleVae;
      const vae = await session(m, 'vae_decoder', device, opts.onProgress, p.signal);
      const img = (await vae.run({ latent_sample: new ort.Tensor('float32', z, [1, 4, Lh, Lw]) })).sample;
      if (lowMemory) await releaseSession(m, 'vae_decoder', device);
      const [, , H, W] = img.dims as number[];
      const d = img.data as Float32Array;
      const rgb = new Uint8ClampedArray(H * W * 3);
      for (let y = 0; y < H; y++)
        for (let xx = 0; xx < W; xx++) for (let c = 0; c < 3; c++) rgb[(y * W + xx) * 3 + c] = Math.round((d[c * H * W + y * W + xx] / 2 + 0.5) * 255);
      return { rgb, width: W, height: H, info };
    } catch (err) {
      throw toEdgewiseError(err, 'Painting with sd-turbo');
    }
  });
}
