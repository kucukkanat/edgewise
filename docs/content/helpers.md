---
title: Helpers
group: Extras
order: 1
eyebrow: edgewise/helpers
---

# Helpers

`edgewise/helpers` holds one-line shortcuts built on the six verbs. Each is a few lines of code you could write yourself; read the source when you need something slightly different.

```ts
import { caption, route, vectorIndex } from 'edgewise/helpers';
```

## Built on generate

| Helper | Does |
| --- | --- |
| `transcribe(audio, opts?)` | speech to text with `stt:realtime` |
| `caption(image, { detail })` | a caption with Florence-2; `detail`: `'short'`, `'detailed'`, `'more-detailed'` |
| `ocr(image)` | text and regions with Florence-2 |
| `detect(image, find?)` | object boxes, or boxes for a phrase |
| `watch(video, { prompt, everyMs, onResult })` | ask about a live video every few seconds; returns `stop()` |

## Built on evaluate

| Helper | Does |
| --- | --- |
| `route(text, lanes, { model, threshold, otherwise })` | picks one lane |
| `classify(text, model)` | the label of a fixed-label classifier |
| `lint(text, policies)` | `{ passed, findings, answers }`; every policy needs a threshold |
| `redact(text, { model, types, threshold, mask })` | masks personal data |
| `replaceSpans(text, spans, mask)` | replaces non-overlapping spans |

## Built on embed

| Helper | Does |
| --- | --- |
| `cosine(a, b)` | cosine similarity |
| `vectorIndex({ name, model, dimensions, gpu })` | a persistent exact k-NN index with `add`, `remove`, `search`, `query`, `pairs`, `save`, `clear` |

## Built on forecast and audio

| Helper | Does |
| --- | --- |
| `detectAnomalies(series, opts)` | values outside the expected range |
| `detectSpeech(audio, vadOptions)` | speech segments in a recording, in seconds |
