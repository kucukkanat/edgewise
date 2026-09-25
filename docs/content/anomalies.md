---
title: Anomaly detection
group: Guides
order: 6
---

# Anomaly detection

## The idea

Forecast each value from the values before it. If the real value falls well outside the predicted range, it is unusual. Chronos-Bolt is zero-shot, so this works on a new series without training.

## The helper

```ts
import { detectAnomalies } from 'edgewise/helpers';

const found = await detectAnomalies(latencyMs, { warmup: 96 });
// [{ index: 140, value: 912, expected: [180, 260], direction: 'high' }]
```

| Option | Default | |
| --- | --- | --- |
| `range` | `[0.1, 0.9]` | quantiles bounding the expected range |
| `tolerance` | 0.5 | margin on each side, as a fraction of the range's width |
| `warmup` | 64 | values used before the first check |
| `window` | 512 | history used for each forecast |
| `model` | `'forecast:default'` | a forecast model |

Checks are batched sixteen at a time, so a series of a few thousand points takes seconds.

## Live data

For a stream, keep a rolling buffer and forecast one step ahead as each value arrives:

```ts
import { forecast } from 'edgewise';

const next = await forecast({ model: 'forecast:default', series: buffer.slice(-512), horizon: 1, quantiles: [0.05, 0.95] });
const [lo, hi] = [next.quantiles[0.05][0], next.quantiles[0.95][0]];
if (value < lo || value > hi) alert(value);
```

## Tips

- A range of `[0.1, 0.9]` holds about 80% of normal values. The `tolerance` margin keeps ordinary noise from being flagged; lower it to catch smaller deviations.
- Give the model at least a few seasonal cycles of history (for hourly data with a daily pattern, 72 values or more).
- Resample uneven data first: `forecast` accepts `{ t, v }` points and does it for you.
