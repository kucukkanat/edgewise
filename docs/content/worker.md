---
title: Worker mode
group: Guides
order: 8
description: Run Edgewise in a Web Worker so model loading and inference never block the page.
---

# Worker mode

Loading a model and running it are heavy. In the browser they compete with your UI for the main thread. `edgewise/worker` moves all of it into a Web Worker, with the same API.

## Set it up

Create a worker file:

```ts file=edgewise.worker.ts
import { serveWorker } from 'edgewise/worker';

serveWorker();
```

Connect to it from the page:

```ts file=main.ts
import { connectWorker } from 'edgewise/worker';

const ew = connectWorker(new Worker(new URL('./edgewise.worker.ts', import.meta.url), { type: 'module' }));

const run = ew.generate({ model: 'text:default', input: 'Write a haiku about the sea.' });
for await (const delta of run) output.textContent += delta;
```

Vite, webpack 5, Rspack, Parcel and esbuild all bundle `new Worker(new URL(…, import.meta.url))` as a separate module.

## What crosses the boundary

Every verb works through the worker with the same options and results:

| | How |
| --- | --- |
| Streams and `run.events` | forwarded chunk by chunk |
| `cancel()` and `signal` | the page rejects at once and tells the worker to stop |
| `onProgress` | forwarded as load events |
| `tools` | the model runs in the worker; each tool's `execute` runs **on the page**, after validating its input with your schema |
| `approve` | asked on the page |
| `schema` | sent as JSON Schema; the final object is validated with your Zod schema on the page |
| Images | elements, canvases and video frames are converted to pixels on the page; `Blob`, `ImageBitmap` and `ImageData` are sent as they are |
| Video parts | frames are sampled on the page |
| `mic()` and `audioSource()` | utterances are streamed to the worker |
| A running `generate()` as `speak()` input | text deltas are streamed to the worker |
| `SpeechAudio`, `GeneratedImage` | rebuilt on the page, so `.play()`, `toWav()` and `toBlob()` work |

With a schema, the worker checks the JSON loosely and the page checks it strictly, so a mismatch throws `SchemaValidationError` without the in-worker retry.

## Configure the worker

The worker has its own registry and configuration. Configure it through the connection, or in the worker file:

```ts
await ew.configure({ allowPreview: true, maxLoadedModels: 2 });
await ew.preload(['text:default'], { onProgress });
```

```ts file=edgewise.worker.ts
import { defineModel } from 'edgewise';
import { serveWorker } from 'edgewise/worker';

defineModel({ id: 'my-classifier', /* … */ });
serveWorker();
```

## Other runtimes

Bun supports Web Workers, so the same code works there. Node has no Web Worker API; `serveWorker(port)` and `connectWorker(port)` also accept any `MessagePort`, such as one from `worker_threads`.

```ts
ew.terminate(); // stop the worker; pending runs reject with AbortError
```
