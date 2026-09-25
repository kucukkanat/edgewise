import { platform } from '#platform';
import type { Platform } from '../platform/types.ts';
import { debug, getConfig } from './config.ts';
import { AbortError, UnsupportedDeviceError } from './errors.ts';
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
    const mobile = platform.isBrowser && typeof navigator !== 'undefined' && /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
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
  if (m.task === 'chrome-prompt') return c.builtinAI !== 'unavailable';
  if (m.requires?.webgpu && !c.webgpu) return false;
  const cpu: Device = c.runtime === 'browser' || c.runtime === 'worker' ? 'wasm' : 'cpu';
  return m.variants.some((v) => v.devices.includes(cpu) || (c.webgpu && v.devices.includes('webgpu')));
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
  for (const device of wanted) {
    if (opts.dtype) {
      const v = m.variants.find((x) => x.devices.includes(device)) ?? m.variants[0];
      return { device, dtype: opts.dtype, variant: v };
    }
    const v = m.variants.find((x) => x.devices.includes(device) && (!x.shaderF16 || caps.shaderF16));
    if (v) return { device, dtype: v.dtype, variant: v };
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
}

const loaded = new Map<string, Entry>();

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
    return existing.value as Promise<T>;
  }
  const value = loader(ctx);
  const entry: Entry = { key, id: ctx.manifest.id, value, dispose: dispose as Entry['dispose'], lastUsed: Date.now() };
  loaded.set(key, entry);
  value.then(
    () => ctx.progress({ type: 'ready', model: ctx.manifest.id, device: ctx.selection.device, dtype: dtypeLabel(ctx.selection.dtype) }),
    () => loaded.delete(key),
  );
  await evict();
  return value;
}

async function evict(): Promise<void> {
  const max = getConfig().maxLoadedModels;
  if (loaded.size <= max) return;
  const sorted = [...loaded.values()].sort((a, b) => a.lastUsed - b.lastUsed);
  for (const e of sorted.slice(0, loaded.size - max)) {
    loaded.delete(e.key);
    try {
      const v = await e.value;
      await e.dispose?.(v);
    } catch {
      // failed loads have nothing to dispose
    }
    debug('unloaded', e.key);
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
