import { ConfigError } from './errors.ts';

export interface FallbackPolicy {
  /** When the chosen variant cannot run on this device: try the next rung of the device ladder. */
  onUnsupported?: 'next-variant' | 'throw';
  /** When the GPU backend fails at runtime: rerun on the CPU path. */
  onBackendError?: 'cpu' | 'wasm' | 'throw';
}

export interface EdgewiseConfig {
  /** Base URL model files are downloaded from. Default `https://huggingface.co`. */
  hub: string;
  /** Server only: where model files are cached. Default `$EDGEWISE_CACHE` or `~/.cache/edgewise`. */
  cacheDir?: string;
  /**
   * Browser only: where ONNX Runtime loads its `.wasm` files from. Defaults to the jsDelivr CDN.
   * Point it at your own copy of `onnxruntime-web/dist/` to self-host.
   */
  wasmPaths?: string;
  /** Maximum models kept loaded at once. Least recently used models are unloaded first. */
  maxLoadedModels: number;
  /**
   * Memory that loaded models may use together, in bytes. Before a load, least recently used models are
   * unloaded until the new one fits. `'auto'` (default) picks a budget from the device: about 2 GB on
   * phones and low-memory devices (where a model that alone exceeds it throws `OutOfMemoryError`), and a
   * soft limit elsewhere. A number is a strict budget. `'off'` disables it. Sizes are estimates; see
   * `memoryUsage()`.
   */
  memoryBudget: number | 'auto' | 'off';
  /**
   * With `device: 'auto'`, choose a variant on another device when it needs at most half the memory,
   * for example speech models on WebAssembly instead of WebGPU. `'auto'` (default) turns it on for
   * phones and devices with 4 GB of memory or less.
   */
  preferLowMemory: boolean | 'auto';
  fallback: Required<FallbackPolicy>;
  /** Only allow models whose licence is in this list. Empty means allow all. */
  licenses: string[];
  /** Allow preview and experimental models everywhere. */
  allowPreview: boolean;
  /** Server only: use a GPU through vgpu when a hardware adapter is present. */
  serverGpu: 'auto' | 'off' | 'force';
  /** Log what Edgewise decides (device, dtype, fallbacks). */
  debug: boolean;
  speak: { onSynthesize?: (info: { audio: Float32Array; sampleRate: number; voice: unknown }) => void };
  experimental: { webnn?: boolean };
}

const defaults = (): EdgewiseConfig => ({
  hub: 'https://huggingface.co',
  cacheDir: undefined,
  wasmPaths: undefined,
  maxLoadedModels: 4,
  memoryBudget: 'auto',
  preferLowMemory: 'auto',
  fallback: { onUnsupported: 'next-variant', onBackendError: 'cpu' },
  licenses: [],
  allowPreview: false,
  serverGpu: 'auto',
  debug: false,
  speak: {},
  experimental: {},
});

let current: EdgewiseConfig = defaults();
const listeners = new Set<(c: EdgewiseConfig) => void>();

export type ConfigureOptions = Partial<Omit<EdgewiseConfig, 'fallback'>> & { fallback?: FallbackPolicy };

/** Change global settings. Unset fields keep their current value. */
export function configure(options: ConfigureOptions): void {
  if (options.maxLoadedModels !== undefined && (!Number.isInteger(options.maxLoadedModels) || options.maxLoadedModels < 1)) {
    throw new ConfigError('maxLoadedModels must be a positive integer.');
  }
  const b = options.memoryBudget;
  if (b !== undefined && b !== 'auto' && b !== 'off' && !(typeof b === 'number' && b > 0)) {
    throw new ConfigError(`memoryBudget must be a positive number of bytes, 'auto' or 'off', got ${JSON.stringify(b)}.`);
  }
  const p = options.preferLowMemory;
  if (p !== undefined && typeof p !== 'boolean' && p !== 'auto') {
    throw new ConfigError(`preferLowMemory must be true, false or 'auto', got ${JSON.stringify(p)}.`);
  }
  if (options.hub !== undefined && !/^https?:\/\//.test(options.hub)) {
    throw new ConfigError(`hub must be an http(s) URL, got "${options.hub}".`);
  }
  current = {
    ...current,
    ...options,
    fallback: { ...current.fallback, ...(options.fallback ?? {}) },
    speak: { ...current.speak, ...(options.speak ?? {}) },
    experimental: { ...current.experimental, ...(options.experimental ?? {}) },
  };
  for (const l of listeners) l(current);
}

export function getConfig(): Readonly<EdgewiseConfig> {
  return current;
}

/** Test helper: restore defaults. */
export function resetConfig(): void {
  current = defaults();
  for (const l of listeners) l(current);
}

export function onConfigChange(fn: (c: EdgewiseConfig) => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function debug(...args: unknown[]): void {
  if (current.debug) console.debug('[edgewise]', ...args);
}
