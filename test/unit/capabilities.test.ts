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
    expect(sel.dtype).toBe(typeof window !== 'undefined' ? 'fp16' : 'q4');
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
});
