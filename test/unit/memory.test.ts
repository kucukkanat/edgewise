import { configure, resetConfig } from '../../src/core/config.ts';
import { OutOfMemoryError } from '../../src/core/errors.ts';
import { chooseCandidate, estimateMemory, resolveBudget, resolvePreferLowMemory } from '../../src/core/memory.ts';
import { registry } from '../../src/core/registry.ts';
import { type LoadContext, loadCached, loadedModels, memoryUsage, unload } from '../../src/core/runtime.ts';
import type { Device, Manifest, Variant } from '../../src/core/types.ts';

const GB = 1e9;
const GiB = 2 ** 30;

afterEach(async () => {
  resetConfig();
  await unload();
});

describe('estimateMemory', () => {
  it('uses measured memory, for all devices or one', () => {
    expect(estimateMemory({ dtype: 'q8', devices: ['wasm'], memory: 5 }, 'wasm')).toBe(5);
    const v: Variant = { dtype: 'q8', devices: ['wasm', 'cpu'], bytes: 100e6, memory: { wasm: 7 } };
    expect(estimateMemory(v, 'wasm')).toBe(7);
    expect(estimateMemory(v, 'cpu')).toBeGreaterThan(100e6);
  });
  it('estimates from the download size, larger for fp16 on WebAssembly', () => {
    const q4 = estimateMemory({ dtype: 'q4', devices: ['wasm'], bytes: 300e6 }, 'wasm');
    const f16 = estimateMemory({ dtype: 'fp16', devices: ['wasm'], bytes: 300e6 }, 'wasm');
    expect(f16).toBeGreaterThan(q4 * 1.5);
    expect(estimateMemory({ dtype: { a: 'q4', b: 'fp16' }, devices: ['wasm'], bytes: 300e6 }, 'wasm')).toBe(f16);
    expect(estimateMemory({ dtype: 'q4', devices: ['wasm'] }, 'wasm')).toBe(0);
  });
});

describe('budget', () => {
  it('is strict and small on phones, soft on desktops and servers', () => {
    const cfg = { memoryBudget: 'auto' as const };
    expect(resolveBudget(cfg, { runtime: 'browser', mobile: true, memory: null })).toEqual({ bytes: 2 * GB, strict: true });
    expect(resolveBudget(cfg, { runtime: 'browser', mobile: false, memory: 8 * GiB })).toEqual({ bytes: 8 * GB, strict: false });
    expect(resolveBudget(cfg, { runtime: 'browser', mobile: false, memory: 4 * GiB }).strict).toBe(true);
    expect(resolveBudget(cfg, { runtime: 'bun', mobile: false, memory: 16 * GB })).toEqual({ bytes: 12 * GB, strict: false });
  });
  it('can be set or turned off', () => {
    expect(resolveBudget({ memoryBudget: 3e9 }, { runtime: 'bun', mobile: false, memory: null })).toEqual({ bytes: 3e9, strict: true });
    expect(resolveBudget({ memoryBudget: 'off' }, { runtime: 'browser', mobile: true, memory: null }).bytes).toBe(Number.POSITIVE_INFINITY);
    expect(() => configure({ memoryBudget: -1 })).toThrow();
    expect(() => configure({ preferLowMemory: 'yes' as never })).toThrow();
  });
  it('prefers low memory on phones and small devices', () => {
    expect(resolvePreferLowMemory({ preferLowMemory: 'auto' }, { mobile: true, memory: null })).toBe(true);
    expect(resolvePreferLowMemory({ preferLowMemory: 'auto' }, { mobile: false, memory: 4 * GiB })).toBe(true);
    expect(resolvePreferLowMemory({ preferLowMemory: 'auto' }, { mobile: false, memory: 8 * GiB })).toBe(false);
  });
});

describe('chooseCandidate', () => {
  const kokoro = registry.get('kokoro-82m');
  const lfm = registry.get('lfm2.5-350m');
  const cands = (m: Manifest, devices: Device[]) =>
    devices.flatMap((device) => m.variants.filter((v) => v.devices.includes(device) && !v.shaderF16).map((variant) => ({ device, variant })));
  const open = { bytes: Number.POSITIVE_INFINITY, strict: false };

  it('keeps the fastest variant by default', () => {
    expect(chooseCandidate(cands(kokoro, ['webgpu', 'wasm']), open, false)?.device).toBe('webgpu');
  });
  it('moves speech to WebAssembly when it saves at least half the memory', () => {
    expect(chooseCandidate(cands(kokoro, ['webgpu', 'wasm']), open, true)?.device).toBe('wasm');
    expect(chooseCandidate(cands(lfm, ['webgpu', 'wasm']), open, true)?.device).toBe('webgpu');
  });
  it('picks the fastest variant that fits the budget', () => {
    expect(chooseCandidate(cands(kokoro, ['webgpu', 'wasm']), { bytes: 1 * GB, strict: true }, false)?.device).toBe('wasm');
  });
});

describe('loaded-model budget', () => {
  const fake = (id: string, memory: number): LoadContext => {
    const variant: Variant = { dtype: 'q8', devices: ['cpu'], memory };
    return { manifest: { id } as Manifest, selection: { device: 'cpu', dtype: 'q8', variant }, progress: () => {} };
  };
  const load = (id: string, memory: number, disposed: string[] = []) =>
    loadCached(
      id,
      fake(id, memory),
      async () => id,
      () => void disposed.push(id),
    );

  it('unloads least recently used models until a new one fits', async () => {
    configure({ memoryBudget: 3 * GB });
    const disposed: string[] = [];
    await load('a', 1 * GB, disposed);
    await load('b', 1 * GB, disposed);
    await load('a', 1 * GB, disposed); // a is now the most recently used
    await load('c', 1.5 * GB, disposed);
    expect(disposed).toEqual(['b']);
    expect(loadedModels().sort()).toEqual(['a', 'c']);
    const u = await memoryUsage();
    expect(u.estimated).toBe(2.5 * GB);
    expect(u.budget).toBe(3 * GB);
    expect(u.models.map((m) => m.id).sort()).toEqual(['a', 'c']);
  });
  it('throws OutOfMemoryError when one model exceeds a strict budget', async () => {
    configure({ memoryBudget: 1 * GB });
    await expect(load('big', 2 * GB)).rejects.toBeInstanceOf(OutOfMemoryError);
    configure({ memoryBudget: 'off' });
    await expect(load('big', 2 * GB)).resolves.toBe('big');
  });
});
