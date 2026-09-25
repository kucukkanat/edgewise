import { detectAnomalies } from '../../src/helpers/index.ts';
import { forecast } from '../../src/index.ts';
import { it2, on, report, suite } from './setup.ts';

const all = ['bun', 'node', 'browser'];
const wave = (n: number, from = 0) => Array.from({ length: n }, (_, i) => 50 + 20 * Math.sin(((from + i) / 24) * 2 * Math.PI));

suite('forecast · real models', () => {
  for (const model of ['chronos-bolt-tiny', 'chronos-bolt-small']) {
    it2(on(all, `continues a seasonal series (${model})`), async () => {
      const r = await forecast({ model, series: wave(240), horizon: 24 });
      report(model, r.info);
      const truth = wave(24, 240);
      const mae = truth.reduce((a, v, i) => a + Math.abs(v - r.median[i]), 0) / 24;
      expect(mae).toBeLessThan(4);
      for (let i = 0; i < 24; i++) {
        expect(r.quantiles[0.1][i]).toBeLessThanOrEqual(r.median[i] + 1e-3);
        expect(r.quantiles[0.9][i]).toBeGreaterThanOrEqual(r.median[i] - 1e-3);
      }
    });
  }

  it2(on(all, 'forecasts past the model horizon and several series at once'), async () => {
    const r = await forecast({ model: 'chronos-bolt-tiny', series: [wave(200), wave(200).map((x) => x * 2)], horizon: 96, quantiles: [0.1, 0.25, 0.9] });
    expect(r.forecasts).toHaveLength(2);
    expect(r.forecasts[1].median).toHaveLength(96);
    expect(Object.keys(r.forecasts[0].quantiles).map(Number).sort()).toEqual([0.1, 0.25, 0.9]);
  });

  it2(on(all, 'flags an injected spike as an anomaly'), async () => {
    const s = wave(160);
    s[140] += 60;
    const found = await detectAnomalies(s, { warmup: 96 });
    expect(found.map((a) => a.index)).toContain(140);
  });
});
