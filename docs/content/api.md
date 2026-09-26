---
title: API reference
group: Reference
order: 20
---

# API reference

The full reference, generated from the source with TypeDoc, is at **[API reference →](api/index.html)**.

## Entry points

| Import | Contains |
| --- | --- |
| `edgewise` | the six verbs, `mic`, `audioSource`, `tool`, question builders, `configure`, `capabilities`, `registry`, `defineModel`, `preload`, `unload`, errors |
| `edgewise/helpers` | [helpers](helpers) built on the verbs |
| `edgewise/react` | [React hooks](react) |
| `edgewise/ai-sdk` | [Vercel AI SDK providers](aisdk) |
| `edgewise/test` | [mock models](testing) |
| `edgewise/worker` | [worker mode](worker): `serveWorker`, `connectWorker` |

## configure()

| Option | Default | |
| --- | --- | --- |
| `hub` | `https://huggingface.co` | where model files download from |
| `cacheDir` | `$EDGEWISE_CACHE` or `~/.cache/edgewise` | servers |
| `wasmPaths` | jsDelivr | browsers: where ONNX Runtime loads `.wasm` from |
| `maxLoadedModels` | 4 | models kept in memory |
| `allowPreview` | false | allow preview and experimental models |
| `licenses` | `[]` (all) | licence allow-list |
| `serverGpu` | `'auto'` | `'auto'`, `'off'` or `'force'` |
| `fallback` | see [Errors](errors) | |
| `debug` | false | log decisions |

## Lifecycle

```ts
import { cache, persist, preload, storage, unload, loadedModels } from 'edgewise';

await preload(['text:default'], { onProgress });   // download and compile ahead of time
await persist();                                 // browsers: ask to keep the cache when space is low
await storage();                                 // { quota, usage }
await cache.list();                              // cached files
await cache.delete('lfm2.5-350m');               // remove a model's files
await unload();                                  // release all loaded models
```
