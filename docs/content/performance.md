---
title: Performance
group: Guides
order: 9
description: Make on-device models fast and keep them within memory: WebGPU, WebAssembly threads and cross-origin isolation, model sizes, workers and phones.
---

# Performance

The same model can be several times faster or slower depending on where it runs. This page explains the differences and what you can change.

## Where models run fastest

Measured on one 2-core machine with no GPU. The model had already loaded; one run each.

| | Bun (ONNX Runtime for Node) | Browser, WebAssembly, one thread |
| --- | --- | --- |
| `lfm2.5-350m`, first token | 155 ms | 1,072 ms |
| `lfm2.5-350m`, tokens per second | 26 | 4.2 |
| `kokoro-82m`, speech speed | 1.0× real time | 0.4× real time |

The server build uses the native ONNX Runtime library: all cores, full CPU vector instructions, and 4-bit weights. A browser without WebGPU runs a WebAssembly build of ONNX Runtime, which is slower. Without cross-origin isolation (below) it also runs on a single thread.

A browser with a hardware GPU runs models on WebGPU instead, which is usually the fastest option in a browser. Check what a device can do:

```ts
import { capabilities } from 'edgewise';

const c = await capabilities();
c.webgpu;       // WebGPU is available
c.hardwareGpu;  // …on a real GPU, not a software adapter like SwiftShader
c.threads;      // WebAssembly can use several threads (needs cross-origin isolation)
```

Every result also reports what ran: `info.device` is `'webgpu'`, `'wasm'` or `'cpu'`.

## WebAssembly threads need cross-origin isolation

Browsers only allow the `SharedArrayBuffer` that WebAssembly threads need when the page is **cross-origin isolated**. Without it, ONNX Runtime Web falls back to one thread, which on multi-core machines is several times slower on the WebAssembly path. WebGPU is not affected.

A page is cross-origin isolated when it is served with these two headers:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

`require-corp` also blocks cross-origin resources that do not opt in. Model files from Hugging Face are fetched with CORS, so they keep working, but third-party images, iframes and scripts without CORS or `Cross-Origin-Resource-Policy` headers stop loading. `Cross-Origin-Embedder-Policy: credentialless` is a gentler alternative in Chromium browsers: cross-origin requests without credentials are allowed. Check `crossOriginIsolated` in the console after deploying.

How to set the headers:

| Host | Where |
| --- | --- |
| Vite dev server | `server.headers` in `vite.config.ts` |
| Netlify, Cloudflare Pages | a `_headers` file |
| Vercel | `headers` in `vercel.json` |
| Nginx | `add_header` in the server block |
| Bun | set them on the `Response` in `Bun.serve()` |

```ts file=vite.config.ts
export default {
  server: { headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' } },
};
```

### GitHub Pages cannot set these headers

GitHub Pages does not let you set response headers, so pages hosted there are never cross-origin isolated, and WebAssembly runs on one thread. This site is on GitHub Pages, so its live demos run single-threaded on devices without WebGPU; your own app on a host that sets the headers will be faster.

A common workaround is a service worker that adds the headers to every response, such as [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker). It needs one reload on the first visit (the worker has to be installed before it can change responses), and it does not work in every setting, for example in some private browsing modes. If you control the host, prefer real headers.

## Model size and memory

- **Pick the smallest model that does the job.** `text:tiny` (`lfm2.5-230m`) and `stt:tiny` (`moonshine-tiny`) start faster and use less memory than their larger siblings. For judging, routing and PII, `evaluate` with an encoder takes milliseconds and far less memory than a language model.
- **Memory is larger than the download.** A model's weights are held in memory, often in a wider format than the file, plus the runtime's working memory. For example, in a WebAssembly browser tab `lfm2.5-350m` uses about 3 GB: its 4-bit exports use an operator ONNX Runtime Web's WebAssembly build lacks, so it runs from fp16 weights that are expanded to fp32. On WebGPU and on servers it runs 4-bit and uses much less.
- **Phones are strict.** iOS closes a tab that uses too much memory, and Safari then shows "A problem repeatedly occurred". Prefer WebGPU there, keep to small models, run speech models (`moonshine-tiny`, `kokoro-82m`) on WebAssembly, and load only what the current screen needs.
- **Unload what you are done with.** `unload(id)` frees a model, and `configure({ maxLoadedModels })` (default 4) caps how many stay loaded; the least recently used one is freed first.

```ts
import { configure, unload } from 'edgewise';

configure({ maxLoadedModels: 2 });
await unload('lfm2.5-vl-450m');
```

## Loading

- **Preload early.** Call `preload(['text:default'], { onProgress })` while the user is still reading, so the first request does not wait for a download.
- **Downloads are cached.** Files are stored once (Cache Storage in browsers, the cache folder on servers). Ask the browser to keep them with `persist()`, so it does not evict them under storage pressure.
- **Compiling takes a moment.** The first run on a device compiles the model's graph. Later runs in the same session are fast.

## Keeping the page responsive

WebAssembly inference runs on the thread that calls it. On the main thread, a long `generate()` blocks rendering and input, and, if you use a microphone, voice activity detection. [Worker mode](worker) moves models into a Web Worker with the same API:

```ts
import { connectWorker } from 'edgewise/worker';

const ew = connectWorker(new Worker(new URL('./edgewise.worker.ts', import.meta.url), { type: 'module' }));
for await (const delta of ew.generate({ model: 'text:default', input: 'Hi' })) out.append(delta);
```

A worker starts a second copy of ONNX Runtime. On phones, where memory is tight, one runtime on the main thread may be the better trade.

## Getting output sooner

- **Stream.** Iterate a Run to show tokens as they arrive; time to first token matters more to users than total time.
- **Speak while generating.** Pass a running `generate()` to `speak()`: it speaks the first sentence while the model writes the next.
- **Keep replies short.** Set `maxTokens`, and ask for brevity in the system prompt. Generation time grows with every token.
- **Batch embeddings.** `embed({ values: [...] })` runs many texts in batches, which is much faster than one call per text.
- **Cancel what you no longer need.** `run.cancel()`, an `AbortSignal`, or leaving a `for await` loop stops the work at once and frees the device for the next request.

## Servers

On Bun and Node, Edgewise uses ONNX Runtime's native library on the CPU, and vgpu for WebGPU on a GPU machine. For batch jobs, APIs and desktop apps this is the fastest place to run models. See [Runtimes and GPUs](runtimes).
