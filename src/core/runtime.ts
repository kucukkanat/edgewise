import { platform } from '#platform';
import type { Platform } from '../platform/types.ts';
import { debug, getConfig } from './config.ts';
import { AbortError, OutOfMemoryError, UnsupportedDeviceError } from './errors.ts';
import { type Candidate, chooseCandidate, estimateMemory, resolveBudget, resolvePreferLowMemory } from './memory.ts';
import { setRunnableCheck } from './registry.ts';
import type { Capabilities, CommonOptions, Device, Dtype, LoadEvent, Manifest, Variant } from './types.ts';

export function getPlatform(): Platform {
  return platform;
}

let capsPromise: Promise<Capabilities> | null = null;
let capsCache: Capabilities | null = null;

async function builtinAI(): Promise<Capabilities['builtinAI']> {
  const LM = (globalThis as { LanguageModel?: { availability?: () => Promise<string> } }).LanguageModel;
  if (!LM?.availability) return 'unavailable';
  try {
    const a = await LM.availability();
    if (a === 'available' || a === 'readily') return 'available';
    if (a === 'downloadable' || a === 'downloading' || a === 'after-download') return 'downloadable';
    return 'unavailable';
  } catch {
    return 'unavailable';
  }
}

/** What this device can do. Cached after the first call. */
export function capabilities(): Promise<Capabilities> {
  capsPromise ??= (async () => {
    const [gpu, storage, ai] = await Promise.all([platform.detectGpu(), platform.storageEstimate(), builtinAI()]);
    const nav = typeof navigator !== 'undefined' ? (navigator as Navigator & { maxTouchPoints?: number }) : undefined;
    // iPadOS reports a Mac user agent; touch support tells them apart.
    const mobile =
      platform.isBrowser &&
      !!nav &&
      (/Android|iPhone|iPad|iPod|Mobile/i.test(nav.userAgent) || (/Macintosh/.test(nav.userAgent) && (nav.maxTouchPoints ?? 0) > 1));
    const tier: Capabilities['tier'] = !gpu.webgpu || !gpu.hardware ? 'cpu' : mobile ? 'mobile' : gpu.shaderF16 && gpu.hardware ? 'gpu-high' : 'gpu';
    const caps: Capabilities = {
      runtime: platform.name,
      webgpu: gpu.webgpu,
      shaderF16: gpu.shaderF16,
      adapter: gpu.adapter,
      hardwareGpu: gpu.hardware,
      gpuProvider: gpu.provider,
      crossOriginIsolated: platform.crossOriginIsolated(),
      threads: platform.isBrowser ? platform.crossOriginIsolated() && typeof SharedArrayBuffer !== 'undefined' : true,
      builtinAI: ai,
      cores: platform.cores(),
      mobile,
      memory: platform.memory(),
      storage,
      tier,
    };
    capsCache = caps;
    return caps;
  })();
  return capsPromise;
}

/** Reset cached capabilities (tests, or after the user plugs in a GPU). */
export function resetCapabilities(): void {
  capsPromise = null;
  capsCache = null;
}

setRunnableCheck((m) => {
  const c = capsCache;
  if (!c) return true;
  return isRunnable(m, c);
});

export function isRunnable(m: Manifest, c: Capabilities): boolean {
  if (m.requires?.browser && c.runtime !== 'browser' && c.runtime !== 'worker') return false;
  if (m.requires?.server && (c.runtime === 'browser' || c.runtime === 'worker')) return false;
  if (m.task === 'chrome-prompt') return c.builtinAI !== 'unavailable';
  if (m.requires?.webgpu && !c.webgpu) return false;
  const cpu: Device = c.runtime === 'browser' || c.runtime === 'worker' ? 'wasm' : 'cpu';
  return m.variants.some((v) => v.devices.includes(cpu) || (c.webgpu && v.devices.includes('webgpu') && (!v.shaderF16 || c.shaderF16)));
}

export interface Selection {
  device: Device;
  dtype: Dtype | Record<string, Dtype>;
  variant: Variant;
}

export function dtypeLabel(d: Dtype | Record<string, Dtype>): string {
  return typeof d === 'string'
    ? d
    : Object.entries(d)
        .map(([k, v]) => `${k}:${v}`)
        .join(',');
}

/** Pick device and dtype for a model on this device, following the device ladder. */
export async function selectVariant(m: Manifest, opts: Pick<CommonOptions, 'device' | 'dtype'> = {}): Promise<Selection> {
  const caps = await capabilities();
  const browser = platform.isBrowser;
  const cpuDevice: Device = browser ? 'wasm' : 'cpu';
  if (m.requires?.browser && !browser) {
    throw new UnsupportedDeviceError(`"${m.id}" only runs in browsers.`, { hint: `Use another model on ${caps.runtime}.` });
  }
  if (m.requires?.server && browser) {
    throw new UnsupportedDeviceError(`"${m.id}" runs on servers only (Bun or Node); it is too large for a browser tab.`, {
      hint: (m.config?.browserAlternative as string | undefined) ? `Use "${m.config?.browserAlternative}" in browsers.` : 'Use a smaller model in browsers.',
    });
  }
  let wanted: Device[];
  const dev = opts.device ?? 'auto';
  // Software adapters (SwiftShader, llvmpipe) are slower than the CPU path, so 'auto' skips them.
  const gpuOk = caps.webgpu && (caps.hardwareGpu || (!browser && getConfig().serverGpu === 'force'));
  if (dev === 'auto') wanted = gpuOk ? ['webgpu', cpuDevice] : [cpuDevice];
  else if (dev === 'wasm' || dev === 'cpu') wanted = [cpuDevice];
  else {
    if (!caps.webgpu) {
      if (getConfig().fallback.onUnsupported === 'throw') {
        throw new UnsupportedDeviceError(`WebGPU is not available on this device.`, {
          hint: browser
            ? 'This browser has no WebGPU adapter. Use device: "wasm".'
            : 'No hardware GPU was found. On a GPU machine check drivers, or run `npx vgpu doctor`.',
        });
      }
      debug(`"${m.id}": WebGPU requested but unavailable, falling back to ${cpuDevice}`);
      wanted = [cpuDevice];
    } else wanted = ['webgpu'];
  }
  if (m.requires?.webgpu && !caps.webgpu) {
    throw new UnsupportedDeviceError(`"${m.id}" needs WebGPU, which this device does not have.`, {
      hint: browser ? 'Use a browser with WebGPU, or pick a model that runs on WASM.' : 'Run on a machine with a GPU.',
    });
  }
  const candidates: Candidate[] = wanted.flatMap((device) =>
    m.variants.filter((x) => x.devices.includes(device) && (!x.shaderF16 || caps.shaderF16)).map((variant) => ({ device, variant })),
  );
  const cfg = getConfig();
  // An explicit device is the caller's choice; only 'auto' may trade speed for memory across devices.
  const pick = chooseCandidate(candidates, resolveBudget(cfg, caps), dev === 'auto' && resolvePreferLowMemory(cfg, caps));
  if (pick) {
    if (pick !== candidates[0]) debug(`"${m.id}": chose ${pick.device} (${dtypeLabel(pick.variant.dtype)}) to save memory`);
    // A dtype override still has to use a variant that runs on this device.
    return { device: pick.device, dtype: opts.dtype ?? pick.variant.dtype, variant: pick.variant };
  }
  throw new UnsupportedDeviceError(`"${m.id}" has no variant for ${wanted.join(' or ')} on this device.`, {
    hint: `"${m.id}" runs on ${[...new Set(m.variants.flatMap((v) => v.devices))].join(', ')}.`,
  });
}

// ---------------------------------------------------------------- loaded-model cache

interface Entry {
  key: string;
  id: string;
  value: Promise<unknown>;
  dispose?: (v: unknown) => Promise<void> | void;
  lastUsed: number;
  /** Use order; timestamps can tie. */
  seq: number;
  device: Device;
  dtype: string;
  /** Estimated bytes in memory. */
  bytes: number;
}

const loaded = new Map<string, Entry>();
let seq = 0;

export interface LoadContext {
  manifest: Manifest;
  selection: Selection;
  progress: (e: LoadEvent) => void;
  signal?: AbortSignal;
}

/**
 * Load a model once and keep it in an LRU cache. Concurrent calls share one load.
 */
export async function loadCached<T>(
  key: string,
  ctx: LoadContext,
  loader: (ctx: LoadContext) => Promise<T>,
  dispose?: (v: T) => Promise<void> | void,
): Promise<T> {
  const existing = loaded.get(key);
  if (existing) {
    existing.lastUsed = Date.now();
    existing.seq = ++seq;
    return existing.value as Promise<T>;
  }
  const bytes = estimateMemory(ctx.selection.variant, ctx.selection.device);
  const budget = resolveBudget(getConfig(), await capabilities());
  if (budget.strict && bytes > budget.bytes) {
    throw new OutOfMemoryError(
      `"${ctx.manifest.id}" needs about ${gb(bytes)} of memory on ${ctx.selection.device}, more than this device's ${gb(budget.bytes)} budget.`,
      { hint: 'Pick a smaller model or dtype, try another device, or raise the limit with configure({ memoryBudget }).' },
    );
  }
  // Free room before loading, so the old and new models are never in memory together.
  await evict(budget.bytes - bytes, getConfig().maxLoadedModels - 1);
  const again = loaded.get(key);
  if (again) {
    again.lastUsed = Date.now();
    again.seq = ++seq;
    return again.value as Promise<T>;
  }
  const value = loader(ctx);
  const entry: Entry = {
    key,
    id: ctx.manifest.id,
    value,
    dispose: dispose as Entry['dispose'],
    lastUsed: Date.now(),
    seq: ++seq,
    device: ctx.selection.device,
    dtype: dtypeLabel(ctx.selection.dtype),
    bytes,
  };
  loaded.set(key, entry);
  value.then(
    () => ctx.progress({ type: 'ready', model: ctx.manifest.id, device: ctx.selection.device, dtype: dtypeLabel(ctx.selection.dtype) }),
    () => loaded.delete(key),
  );
  return value;
}

function gb(n: number): string {
  return `${(n / 1e9).toFixed(1)} GB`;
}

/** Unload least recently used models until at most `maxCount` remain and they use at most `maxBytes`. */
async function evict(maxBytes: number, maxCount: number): Promise<void> {
  const total = () => [...loaded.values()].reduce((a, e) => a + e.bytes, 0);
  const sorted = [...loaded.values()].sort((a, b) => a.seq - b.seq);
  for (const e of sorted) {
    if (loaded.size <= maxCount && total() <= maxBytes) return;
    loaded.delete(e.key);
    try {
      const v = await e.value;
      await e.dispose?.(v);
    } catch {
      // failed loads have nothing to dispose
    }
    debug('unloaded', e.key, 'to make room');
  }
}

export interface MemoryUsage {
  /** Estimated bytes all loaded models use together. */
  estimated: number;
  /** The budget in effect, in bytes (`Infinity` when off), and whether it is strict. */
  budget: number;
  strict: boolean;
  models: { id: string; device: Device; dtype: string; bytes: number; lastUsed: number }[];
  /** Resident memory of the whole process on servers. `null` in browsers, which do not report it. */
  measured: number | null;
}

/** Loaded models, their estimated memory and the budget. */
export async function memoryUsage(): Promise<MemoryUsage> {
  const budget = resolveBudget(getConfig(), await capabilities());
  const models = [...loaded.values()].map((e) => ({ id: e.id, device: e.device, dtype: e.dtype, bytes: e.bytes, lastUsed: e.lastUsed }));
  return { estimated: models.reduce((a, m) => a + m.bytes, 0), budget: budget.bytes, strict: budget.strict, models, measured: measuredMemory() };
}

function measuredMemory(): number | null {
  if (platform.isBrowser) return null;
  const proc = (globalThis as { process?: { memoryUsage?: () => { rss: number } } }).process;
  try {
    return proc?.memoryUsage?.().rss ?? null;
  } catch {
    return null;
  }
}

/** Unload one model (by ID) or every model. Frees GPU and CPU memory. */
export async function unload(id?: string): Promise<void> {
  for (const e of [...loaded.values()]) {
    if (id && e.id !== id) continue;
    loaded.delete(e.key);
    try {
      const v = await e.value;
      await e.dispose?.(v);
    } catch {
      // ignore
    }
  }
}

export function loadedModels(): string[] {
  return [...new Set([...loaded.values()].map((e) => e.id))];
}

export function checkSignal(signal?: AbortSignal): void {
  if (signal?.aborted) throw new AbortError(undefined, { cause: signal.reason });
}
