import { type ChronosConfig, chronosForecast } from '../backends/chronos.ts';
import { mockForecast } from '../backends/mock.ts';
import { ConfigError } from '../core/errors.ts';
import { resolveManifest } from '../core/registry.ts';
import type { CommonOptions, ModelRef, RunInfo } from '../core/types.ts';

/** Evenly spaced values (oldest first), or timestamped points. `NaN` marks a missing value. */
export type SeriesInput = number[] | Float32Array | { t: Date | number; v: number }[];

export interface ForecastOptions extends CommonOptions {
  model: ModelRef;
  /** One series, or an array of series to forecast in one call. */
  series: SeriesInput | SeriesInput[];
  /** Steps to predict. Up to 64 in one pass; longer horizons repeat the forecast on its own median. */
  horizon: number;
  /** Quantile levels to return, between 0.1 and 0.9. Default [0.1, 0.5, 0.9]. */
  quantiles?: number[];
  /** Use only the most recent N values. */
  context?: number;
}

export interface SeriesForecast {
  /** The 0.5 quantile. */
  median: Float32Array;
  /** Average of the model's quantiles: a good point forecast. */
  mean: Float32Array;
  /** `quantiles[0.1]` is the value the series has a 10% chance of falling below at each step. */
  quantiles: Record<number, Float32Array>;
  /** For timestamped input: the time of each forecast step, in ms since the epoch. */
  times?: number[];
}

export type ForecastResult = SeriesForecast & { info: RunInfo };
export interface ForecastManyResult {
  forecasts: SeriesForecast[];
  info: RunInfo;
}

function isPoints(s: unknown): s is { t: Date | number; v: number }[] {
  return Array.isArray(s) && s.length > 0 && typeof s[0] === 'object' && s[0] !== null && 'v' in (s[0] as object);
}

function isSingle(s: unknown): s is SeriesInput {
  return s instanceof Float32Array || isPoints(s) || (Array.isArray(s) && (s.length === 0 || typeof s[0] === 'number'));
}

/** Resample timestamped points to an even grid. Returns values, the step in ms and the last time. */
export function resamplePoints(points: { t: Date | number; v: number }[]): { values: Float32Array; step: number; last: number } {
  const pts = points.map((p) => ({ t: p.t instanceof Date ? p.t.getTime() : p.t, v: p.v })).sort((a, b) => a.t - b.t);
  if (pts.length < 2) return { values: Float32Array.from(pts.map((p) => p.v)), step: 1, last: pts[0]?.t ?? 0 };
  const diffs = pts
    .slice(1)
    .map((p, i) => p.t - pts[i].t)
    .filter((d) => d > 0)
    .sort((a, b) => a - b);
  const step = diffs[Math.floor((diffs.length - 1) / 2)] || 1;
  const first = pts[0].t;
  const n = Math.round((pts[pts.length - 1].t - first) / step) + 1;
  const values = new Float32Array(n).fill(Number.NaN);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = first + i * step;
    while (j < pts.length - 1 && pts[j + 1].t <= t) j++;
    const a = pts[j];
    const b = pts[j + 1];
    if (Math.abs(a.t - t) < step / 2) values[i] = a.v;
    else if (b && b.t - a.t <= 2 * step) values[i] = a.v + ((b.v - a.v) * (t - a.t)) / (b.t - a.t);
  }
  return { values, step, last: first + (n - 1) * step };
}

function interpolate(levels: number[], qs: Float32Array[], want: number, H: number): Float32Array {
  const exact = levels.indexOf(want);
  if (exact >= 0) return qs[exact];
  let lo = 0;
  while (lo < levels.length - 2 && levels[lo + 1] < want) lo++;
  const w = (want - levels[lo]) / (levels[lo + 1] - levels[lo]);
  const out = new Float32Array(H);
  for (let h = 0; h < H; h++) out[h] = qs[lo][h] * (1 - w) + qs[lo + 1][h] * w;
  return out;
}

/** Predict the next values of one or more time series, with uncertainty ranges. */
export async function forecast(options: ForecastOptions & { series: SeriesInput }): Promise<ForecastResult>;
export async function forecast(options: ForecastOptions & { series: SeriesInput[] }): Promise<ForecastManyResult>;
export async function forecast(options: ForecastOptions): Promise<ForecastResult | ForecastManyResult> {
  const m = resolveManifest(options.model, { verb: 'forecast', allowPreview: options.allowPreview, inputs: ['series'] });
  const c = m.config as unknown as ChronosConfig;
  if (!Number.isInteger(options.horizon) || options.horizon < 1 || options.horizon > 1024) {
    throw new ConfigError('horizon must be an integer between 1 and 1024.');
  }
  const levels = c.quantiles;
  const want = options.quantiles ?? [0.1, 0.5, 0.9];
  for (const q of want) {
    if (q < levels[0] || q > levels[levels.length - 1]) {
      throw new ConfigError(`"${m.id}" forecasts quantiles from ${levels[0]} to ${levels[levels.length - 1]}; ${q} is outside that range.`, {
        hint: 'Use quantiles between 0.1 and 0.9, and widen the band yourself if you need rarer events.',
      });
    }
  }
  const single = isSingle(options.series);
  const list: SeriesInput[] = single ? [options.series as SeriesInput] : (options.series as SeriesInput[]);
  if (!list.length) throw new ConfigError('series is empty.');
  let info: RunInfo | undefined;
  const results: SeriesForecast[] = [];
  for (const s of list) {
    let values: Float32Array;
    let times: number[] | undefined;
    if (isPoints(s)) {
      const r = resamplePoints(s);
      values = r.values;
      times = Array.from({ length: options.horizon }, (_, i) => r.last + (i + 1) * r.step);
    } else values = s instanceof Float32Array ? s : Float32Array.from(s as number[]);
    if (values.length < 2) throw new ConfigError('Each series needs at least 2 values.');
    if (options.context && values.length > options.context) values = values.subarray(values.length - options.context);
    const r = m.task === 'mock' ? mockForecast(m, values, options.horizon) : await chronosForecast(m, options, values, options.horizon);
    info = r.info;
    const qmap: Record<number, Float32Array> = {};
    for (const q of want) qmap[q] = interpolate(levels, r.quantiles, q, options.horizon);
    const mean = new Float32Array(options.horizon);
    for (let h = 0; h < options.horizon; h++) mean[h] = r.quantiles.reduce((a, qv) => a + qv[h], 0) / r.quantiles.length;
    results.push({ median: interpolate(levels, r.quantiles, 0.5, options.horizon), mean, quantiles: qmap, times });
  }
  if (single) return { ...results[0], info: info as RunInfo };
  return { forecasts: results, info: info as RunInfo };
}
