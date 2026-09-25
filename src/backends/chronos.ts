import { dtypeLabel, getPlatform, loadCached, selectVariant } from '../core/runtime.ts';
import type { CommonOptions, Manifest, RunInfo } from '../core/types.ts';
import { serialize } from '../core/util.ts';
import type { OrtModule, OrtSession } from '../platform/types.ts';
import { fetchModelFile } from './files.ts';

export interface ChronosConfig {
  quantiles: number[];
  predictionLength: number;
  contextLength: number;
  patchSize: number;
  file: string;
  parts?: number;
}

function cfg(m: Manifest): ChronosConfig {
  return m.config as unknown as ChronosConfig;
}

/** Instance-normalize, NaN-mask, left-pad and patch a series the way Chronos-Bolt expects. */
export function prepareContext(series: Float32Array, c: Pick<ChronosConfig, 'contextLength' | 'patchSize'>) {
  const x = series.length > c.contextLength ? series.subarray(series.length - c.contextLength) : series;
  let sum = 0;
  let n = 0;
  for (const v of x)
    if (!Number.isNaN(v)) {
      sum += v;
      n++;
    }
  const loc = n ? sum / n : 0;
  let sq = 0;
  for (const v of x) if (!Number.isNaN(v)) sq += (v - loc) ** 2;
  let scale = n ? Math.sqrt(sq / n) : 1;
  if (!Number.isFinite(scale)) scale = 1;
  if (scale === 0) scale = 1e-5;
  const P = c.patchSize;
  const pad = x.length % P ? P - (x.length % P) : 0;
  const L = x.length + pad;
  const patches = L / P;
  const pc = new Float32Array(patches * 2 * P);
  const am = new Float32Array(patches);
  for (let p = 0; p < patches; p++) {
    let observed = 0;
    for (let k = 0; k < P; k++) {
      const idx = p * P + k - pad;
      const v = idx >= 0 ? x[idx] : Number.NaN;
      const ok = !Number.isNaN(v);
      pc[p * 2 * P + k] = ok ? (v - loc) / scale : 0;
      pc[p * 2 * P + P + k] = ok ? 1 : 0;
      if (ok) observed++;
    }
    am[p] = observed > 0 ? 1 : 0;
  }
  return { pc, am, patches, loc, scale };
}

interface Loaded {
  ort: OrtModule;
  session: OrtSession;
}

export async function loadChronos(m: Manifest, opts: CommonOptions): Promise<{ value: Loaded; info: RunInfo }> {
  const sel = await selectVariant(m, { ...opts, device: opts.device === 'webgpu' ? 'webgpu' : 'cpu' });
  const device = getPlatform().isBrowser && sel.device === 'cpu' ? 'wasm' : sel.device;
  const c = cfg(m);
  const value = await loadCached<Loaded>(
    `${m.id}|${device}`,
    { manifest: m, selection: sel, progress: (e) => opts.onProgress?.(e), signal: opts.signal },
    async () => {
      const bytes = await fetchModelFile(m, c.file, { parts: c.parts, signal: opts.signal, onProgress: opts.onProgress });
      const ort = await getPlatform().loadOrt();
      const eps = device === 'webgpu' ? ['webgpu'] : getPlatform().isBrowser ? ['wasm'] : ['cpu'];
      const session = await ort.InferenceSession.create(bytes, { executionProviders: eps });
      return { ort, session };
    },
    async (l) => {
      await l.session.release?.();
    },
  );
  const backend = getPlatform().isBrowser ? `onnxruntime-web:${device}` : `onnxruntime-node:${device}`;
  return { value, info: { model: m.id, device, dtype: dtypeLabel(sel.dtype), backend } };
}

/** Quantile forecasts for one series: `[quantile][step]`, in the series' own units. */
export async function chronosForecast(
  m: Manifest,
  opts: CommonOptions,
  series: Float32Array,
  horizon: number,
): Promise<{ quantiles: Float32Array[]; info: RunInfo }> {
  const c = cfg(m);
  const { value, info } = await loadChronos(m, opts);
  const Q = c.quantiles.length;
  const out = Array.from({ length: Q }, () => new Float32Array(horizon));
  let history = series;
  let produced = 0;
  while (produced < horizon) {
    const { pc, am, patches, loc, scale } = prepareContext(history, c);
    const res = await serialize(`forecast:${m.id}`, () =>
      value.session.run({
        patched_context: new value.ort.Tensor('float32', pc, [1, patches, 2 * c.patchSize]),
        attention_mask: new value.ort.Tensor('float32', am, [1, patches]),
      }),
    );
    const q = res.quantiles.data as Float32Array;
    const H = c.predictionLength;
    const take = Math.min(H, horizon - produced);
    for (let qi = 0; qi < Q; qi++) for (let h = 0; h < take; h++) out[qi][produced + h] = q[qi * H + h] * scale + loc;
    produced += take;
    if (produced < horizon) {
      // Longer horizons: feed the median back in and forecast again.
      const mid = c.quantiles.indexOf(0.5) >= 0 ? c.quantiles.indexOf(0.5) : Math.floor(Q / 2);
      const next = new Float32Array(history.length + take);
      next.set(history);
      next.set(out[mid].subarray(produced - take, produced), history.length);
      history = next;
    }
  }
  return { quantiles: out, info };
}
