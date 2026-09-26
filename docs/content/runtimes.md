---
title: Runtimes and GPUs
group: Get started
order: 4
description: How Edgewise runs in browsers, Bun and Node, and how it uses WebGPU on servers through vgpu.
---

# Runtimes and GPUs

Edgewise is one package with two platform layers, picked by the `browser` export condition.

| | Browser | Bun and Node |
| --- | --- | --- |
| Model runtime | ONNX Runtime Web (via Transformers.js 4) | ONNX Runtime for Node (via Transformers.js 4) |
| GPU | `navigator.gpu` (WebGPU) | vgpu (Dawn, Vulkan / Metal / D3D12) |
| CPU path | WebAssembly (SIMD, threads when [cross-origin isolated](performance#webassembly-threads-need-cross-origin-isolation)) | native CPU |
| Model cache | Cache API | `~/.cache/edgewise` or `$EDGEWISE_CACHE` |
| Audio in | `mic()` | `audioSource()` over any PCM stream |
| Audio out | `.play()` | `toWav()` |
| Images out | `toBlob()`, `toImageData()` | `toPng()` |

## WebGPU on the server with vgpu

On Bun and Node, Edgewise asks [vgpu](https://github.com/vercel/vgpu) for a **hardware** adapter when it first checks capabilities. If one is found:

- `capabilities()` reports `webgpu: true`, `gpuProvider: 'vgpu'` and the adapter name.
- Models request ONNX Runtime's WebGPU execution provider. If that fails to load, Edgewise retries on the CPU and says so in `info`.
- Edgewise's own GPU maths, such as scoring large [vector indexes](search), runs as a WGSL kernel on the vgpu device.

Software renderers (llvmpipe, lavapipe, SwiftShader) are ignored on servers, because the CPU path is faster. The probe device is released right away so your process can exit normally.

```ts
import { configure } from 'edgewise';

configure({ serverGpu: 'auto' });  // default: use a hardware GPU if there is one
configure({ serverGpu: 'off' });   // CPU only
configure({ serverGpu: 'force' }); // accept any adapter, even a software one
```

`EDGEWISE_GPU=off` and `EDGEWISE_GPU=force` do the same from the environment. On Linux, Edgewise skips the probe when no Vulkan driver is installed. Run `npx vgpu doctor` to check a machine.

> [!WARN] The test suite runs on GPU-less CI machines, so server GPU inference is less tested than the CPU path. The vector kernel is tested against the CPU result on a software adapter.

## What is tested where

Real-model tests run each verb on Bun, Node and headless Chromium (WebGPU on SwiftShader, and WASM). The [model list](models) marks a model `stable` only when all three pass. SD-Turbo and Chatterbox are the exceptions: in browsers they need a hardware GPU with enough memory, so the test suite runs them on Bun and Node, and in browsers only when a hardware GPU is present. Both are `preview` until they pass in a browser too.

## Offline and self-hosting

- `configure({ hub })` points every model download, including the Edgewise bucket, at a mirror of huggingface.co.
- `configure({ wasmPaths })` self-hosts ONNX Runtime's WebAssembly files.
- On servers, set `EDGEWISE_CACHE` to a folder you ship with your app and call `preload()` at build time.

## WebAssembly variants

Some 4-bit exports use an ONNX operator (`GatherBlockQuantized`) that ONNX Runtime's WebAssembly build does not include. For those models the registry lists a separate WebAssembly variant (fp16 or 8-bit), which is larger. The [model table](models) shows the smallest download; `info.dtype` shows what was loaded.

## Known limitations

- In browsers, Transformers.js 4.3 serializes model loading, and a load that fails can make later loads on the same page fail with the same error. Reload the page after a failed load.
- Browsers without WebGPU cannot run `paint`, voice cloning, `lfm2.5-1.2b` or `lfm2.5-vl-1.6b`.
- Headless and incognito browsers often have a storage quota under 1 GB. Models still load, but are downloaded again next time.
