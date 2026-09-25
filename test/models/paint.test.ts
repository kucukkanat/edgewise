import { paint } from '../../src/index.ts';
import { it2, on, report, suite } from './setup.ts';

// SD-Turbo ships fp16 weights: it needs WebGPU in the browser, or the CPU backend on servers.
suite('paint · real models', () => {
  it2(on(['bun', 'node'], 'paints a 256×256 image with SD-Turbo'), async () => {
    const steps: number[] = [];
    const run = paint({ model: 'sd-turbo', prompt: 'a red apple on a wooden table, photo', size: '256x256', steps: 1, seed: 42 });
    for await (const s of run) steps.push(s.step);
    const r = await run;
    report('sd-turbo', r.info);
    expect(r.image.width).toBe(256);
    expect(r.image.data.length).toBe(256 * 256 * 4);
    let sum = 0;
    let sq = 0;
    for (let i = 0; i < r.image.data.length; i += 4) {
      sum += r.image.data[i];
      sq += r.image.data[i] ** 2;
    }
    const n = r.image.data.length / 4;
    const std = Math.sqrt(sq / n - (sum / n) ** 2);
    expect(std).toBeGreaterThan(10);
    expect(r.seed).toBe(42);
    const png = await r.image.toPng();
    expect(png.length).toBeGreaterThan(1000);
    expect((await r.image.toBlob()).type).toBe('image/png');
  });
});
