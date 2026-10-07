import { UnsupportedDeviceError } from '../../src/core/errors.ts';
import { registry } from '../../src/core/registry.ts';
import { capabilities, isRunnable, selectVariant } from '../../src/core/runtime.ts';

describe('capabilities', () => {
  it('describes the runtime', async () => {
    const c = await capabilities();
    const expected = typeof window !== 'undefined' ? 'browser' : typeof (globalThis as { Bun?: unknown }).Bun !== 'undefined' ? 'bun' : 'node';
    expect(c.runtime).toBe(expected);
    expect(typeof c.webgpu).toBe('boolean');
    expect(['gpu-high', 'gpu', 'mobile', 'cpu']).toContain(c.tier);
    expect(c.cores === null || c.cores > 0).toBe(true);
  });

  it('picks a CPU variant when there is no GPU', async () => {
    const c = await capabilities();
    const sel = await selectVariant(registry.get('lfm2.5-350m'), { device: 'cpu' });
    expect(sel.device).toBe(typeof window !== 'undefined' ? 'wasm' : 'cpu');
    expect(sel.dtype).toBe('q4');
    if (!c.webgpu) expect((await selectVariant(registry.get('lfm2.5-350m'))).device).not.toBe('webgpu');
  });

  it('honours a forced dtype', async () => {
    const sel = await selectVariant(registry.get('lfm2.5-350m'), { device: 'cpu', dtype: 'fp16' });
    expect(sel.dtype).toBe('fp16');
  });

  it('knows which models this device can run', async () => {
    const c = await capabilities();
    expect(isRunnable(registry.get('embed:tiny'), c)).toBe(true);
    expect(isRunnable(registry.get('chrome:gemini-nano'), { ...c, runtime: 'node' })).toBe(false);
    expect(registry.list({ runnable: true }).length).toBeGreaterThan(5);
  });

  it('keeps server-only models out of browsers', async () => {
    const c = await capabilities();
    const browserGpu = { ...c, runtime: 'browser' as const, webgpu: true, hardwareGpu: true };
    expect(isRunnable(registry.get('judge:large'), browserGpu)).toBe(false);
    expect(isRunnable(registry.get('judge:large'), { ...browserGpu, runtime: 'worker' as const })).toBe(false);
    expect(isRunnable(registry.get('judge:large'), { ...c, runtime: 'node' })).toBe(true);
    if (typeof document !== 'undefined') {
      // Fails before downloading anything, and names the browser model to use instead.
      const err = await selectVariant(registry.get('judge:large')).catch((e: unknown) => e);
      expect(err).toBeInstanceOf(UnsupportedDeviceError);
      expect((err as UnsupportedDeviceError).hint).toMatch(/d1-omni-600m/);
    } else {
      expect((await selectVariant(registry.get('judge:large'))).dtype).toBe('q8');
    }
  });

  it('does not count fp16 WebGPU models as runnable on GPUs without shader-f16', async () => {
    const browserGpu = { ...(await capabilities()), runtime: 'browser' as const, webgpu: true, hardwareGpu: true };
    expect(isRunnable(registry.get('sd-turbo'), { ...browserGpu, shaderF16: true })).toBe(true);
    expect(isRunnable(registry.get('sd-turbo'), { ...browserGpu, shaderF16: false })).toBe(false);
    // Models with a WebAssembly variant still run there.
    expect(isRunnable(registry.get('lfm2.5-vl-450m'), { ...browserGpu, shaderF16: false })).toBe(true);
  });
});
