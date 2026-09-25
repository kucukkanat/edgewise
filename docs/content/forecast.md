---
title: forecast
group: The six verbs
order: 60
eyebrow: verb · → numbers
---

# forecast

Predict the next values of a time series, with uncertainty, using Chronos-Bolt. It is zero-shot: no training on your data.

## Forecast one series

```ts
import { forecast } from 'edgewise';

const { median, quantiles, mean } = await forecast({
  model: 'forecast:default',
  series: hourlyVisitors,   // number[] or Float32Array, oldest first
  horizon: 24,
});

median;          // Float32Array(24)
quantiles[0.1];  // lower edge of the 80% range
quantiles[0.9];  // upper edge
```

## Timestamps and gaps

Pass points with times. Edgewise resamples them onto an even grid (the median gap), fills short gaps, and returns the times of the forecast:

```ts
const r = await forecast({
  model: 'forecast:default',
  series: readings.map((x) => ({ t: x.date, v: x.value })), // Date or ms
  horizon: 12,
});
r.times; // number[] of forecast timestamps (ms)
```

`NaN` values are treated as missing, and the model ignores them.

## Many series at once

```ts
const { forecasts } = await forecast({ model: 'chronos-bolt-small', series: [cpu, memory, disk], horizon: 48 });
forecasts[1].median;
```

## Quantiles, context and horizon

- `quantiles` picks the levels you want, between 0.1 and 0.9. The model predicts nine levels (0.1 to 0.9), and others are interpolated. Default `[0.1, 0.5, 0.9]`.
- `context` limits how much history is used, from the most recent end. Chronos-Bolt reads up to 2048 values.
- `horizon` is 1 to 1024. The model predicts 64 steps at a time; longer horizons are rolled forward on the median, so the range widens faster than the model's own estimate.

## Anomaly detection

`detectAnomalies()` in `edgewise/helpers` forecasts each point from the values before it and flags points outside the predicted range. See [Anomaly detection](anomalies).

## Models

{{models-forecast}}
