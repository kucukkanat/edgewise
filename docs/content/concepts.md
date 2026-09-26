---
title: Core concepts
group: Get started
order: 3
---

# Core concepts

## Verbs

| Verb | Returns | Streams |
| --- | --- | --- |
| `generate` | `text`, `object`, `toolCalls`, `segments` | text deltas, or partial objects with `schema` |
| `evaluate` | typed answers per question | no (it takes milliseconds) |
| `embed` | `Float32Array` vectors | no |
| `speak` | `SpeechAudio` (samples, WAV) | one audio chunk per sentence |
| `paint` | `GeneratedImage` (RGBA, PNG) | a latent preview per step |
| `forecast` | median, mean and quantiles | no |

## Inputs

Every verb takes plain values. `generate` accepts a string, a part, raw media, or an array mixing them. See [Message parts](parts).

## Runs

`generate`, `speak` and `paint` return a `Run`. A Run is both a promise and an async iterable:

```ts
const run = generate({ model: 'text:default', input: 'Hi' });

for await (const delta of run) ui.append(delta); // stream…
const result = await run;                        // …then read the full result

run.cancel();                                    // or stop it
for await (const e of run.events) console.log(e.type); // load, text-delta, tool-call, step, finish
```

A Run starts when you first await or iterate it. It can be iterated once. Chunks are buffered from the moment iteration begins, so a slow consumer never misses output. `signal` and `cancel()` both reject the Run with `AbortError`, and so does leaving a `for await` loop early with `break`: stopping reading stops the model.

## Resolution: models, aliases and the registry

A model reference is an ID (`'lfm2.5-350m'`), an alias (`'text:default'`) or a manifest object. Before downloading anything Edgewise checks that:

1. the model exists and belongs to this verb,
2. it accepts the input types you passed,
3. its status is allowed (`preview` and `experimental` need `allowPreview: true`),
4. its licence is on your allow-list, if you set one with `configure({ licenses })`.

Each failure throws a typed error with a hint. See [Errors](errors).

```ts
import { registry } from 'edgewise';

registry.list({ verb: 'evaluate', status: 'stable' });
registry.alias('judge:router'); // 'nli-deberta-v3-xsmall'
```

## Device ladder

For each model Edgewise tries WebGPU first, then WebAssembly (browser) or CPU (server). It picks the first variant that runs on the device. A variant needing `shader-f16` is skipped when the adapter lacks it. If a GPU run fails at load time, Edgewise retries on the CPU path and records it. `info.device` and `info.backend` tell you what actually ran. Set `device: 'wasm'` or `device: 'webgpu'` to pin it.

## Adding a model

```ts file=my-model.ts
import { defineModel } from 'edgewise';

defineModel({
  id: 'my-classifier',
  verb: 'evaluate',
  accepts: ['text'],
  task: 'sequence-classification',
  source: { repo: 'me/my-classifier-onnx', revision: '3f2a…' }, // always pin a commit
  variants: [{ dtype: 'q8', devices: ['webgpu', 'wasm', 'cpu'], bytes: 70_000_000 }],
  license: 'apache-2.0',
  status: 'preview',
  features: ['label', 'boolean'],
});
```

## Memory

Up to four models stay loaded (`configure({ maxLoadedModels })`). The least recently used one is released first. `unload(id?)` releases models yourself.
