import type { EdgewiseConfig } from './config.ts';
import type { Capabilities, Device, Dtype, Variant } from './types.ts';

const GB = 1e9;

/** Runtime working memory a loaded model adds beyond its weights, by device. */
const OVERHEAD: Record<Device, number> = { webgpu: 0.5 * GB, wasm: 0.25 * GB, cpu: 0.1 * GB };

function widest(dtype: Dtype | Record<string, Dtype>): Dtype {
  const all = typeof dtype === 'string' ? [dtype] : Object.values(dtype);
  for (const d of ['fp32', 'fp16', 'q8', 'int8', 'uint8', 'q4f16', 'q4', 'bnb4'] as Dtype[]) if (all.includes(d)) return d;
  return all[0] ?? 'fp32';
}

/**
 * Bytes a variant is expected to use once loaded on a device: the measured `memory` when the manifest
 * has one, otherwise an estimate from the download size. `0` when neither is known.
 *
 * The estimate reflects how ONNX Runtime holds weights: WebAssembly expands fp16 to fp32 (about 4× the
 * file), WebGPU keeps a copy in GPU buffers, the native CPU runtime maps them nearly as stored.
 */
export function estimateMemory(variant: Variant, device: Device): number {
  const m = variant.memory;
  if (typeof m === 'number') return m;
  if (m?.[device] !== undefined) return m[device] as number;
  if (!variant.bytes) return 0;
  const d = widest(variant.dtype);
  const factor = device === 'webgpu' ? 2 : device === 'wasm' ? (d === 'fp16' ? 4 : d === 'fp32' ? 3 : 2) : 1.5;
  return Math.round(OVERHEAD[device] + variant.bytes * factor);
}

export interface Budget {
  /** Bytes loaded models may use together. `Infinity` when off. */
  bytes: number;
  /** A model that alone exceeds a strict budget throws instead of loading. */
  strict: boolean;
}

/** True on phones and on devices that report 4 GB of memory or less. */
export function lowMemoryDevice(caps: Pick<Capabilities, 'mobile' | 'memory'>): boolean {
  return caps.mobile || (caps.memory !== null && caps.memory <= 4 * 2 ** 30);
}

/** The memory budget in effect for these capabilities and settings. */
export function resolveBudget(cfg: Pick<EdgewiseConfig, 'memoryBudget'>, caps: Pick<Capabilities, 'mobile' | 'memory' | 'runtime'>): Budget {
  const b = cfg.memoryBudget;
  if (b === 'off') return { bytes: Number.POSITIVE_INFINITY, strict: false };
  if (typeof b === 'number') return { bytes: b, strict: true };
  const browser = caps.runtime === 'browser' || caps.runtime === 'worker';
  if (caps.mobile) return { bytes: 2 * GB, strict: true };
  if (browser) {
    // Chromium caps deviceMemory at 8 GiB, so 8 means "8 or more".
    if (caps.memory !== null && caps.memory < 8 * 2 ** 30) return { bytes: Math.max(1.5 * GB, caps.memory * 0.6), strict: lowMemoryDevice(caps) };
    return { bytes: 8 * GB, strict: false };
  }
  return { bytes: caps.memory ? caps.memory * 0.75 : 8 * GB, strict: false };
}

/** Whether to prefer a lower-memory variant over the fastest one. */
export function resolvePreferLowMemory(cfg: Pick<EdgewiseConfig, 'preferLowMemory'>, caps: Pick<Capabilities, 'mobile' | 'memory'>): boolean {
  return cfg.preferLowMemory === 'auto' ? lowMemoryDevice(caps) : cfg.preferLowMemory;
}

export interface Candidate {
  device: Device;
  variant: Variant;
}

/**
 * Pick from candidates listed fastest first. The fastest one wins unless it does not fit the budget
 * (then the fastest that fits), or `preferLow` is on and another needs at most half its memory.
 */
export function chooseCandidate(candidates: Candidate[], budget: Budget, preferLow: boolean): Candidate | undefined {
  const [first] = candidates;
  if (!first) return undefined;
  const est = (c: Candidate) => estimateMemory(c.variant, c.device);
  let pick = first;
  if (est(pick) > budget.bytes) pick = candidates.find((c) => est(c) <= budget.bytes) ?? pick;
  if (preferLow) {
    const smallest = candidates.reduce((a, c) => (est(c) > 0 && est(c) < est(a) ? c : a), pick);
    if (est(smallest) > 0 && est(smallest) <= est(pick) / 2) pick = smallest;
  }
  return pick;
}
