---
title: Quickstart
group: Get started
order: 2
---

# Quickstart

## Install

```bash
bun add edgewise
# or: npm install edgewise · pnpm add edgewise
```

Some features use optional peer dependencies. Install only what you use:

| Package | Needed for |
| --- | --- |
| `zod` | `schema` in `generate()`, and tool inputs |
| `phonemizer` | `speak()` with Kokoro (it bundles espeak-ng, which is GPL-3.0) |
| `react` | `edgewise/react` hooks |
| `ai`, `@ai-sdk/provider` | `edgewise/ai-sdk`, the Vercel AI SDK provider |

## Your first call

```ts
import { generate } from 'edgewise';

const { text, info } = await generate({
  model: 'text:default',                  // alias for lfm2.5-350m; 'text:tiny' is lfm2.5-230m
  input: 'Give me three names for a bakery.',
});
console.log(text, info.device);
```

The first call downloads the model (about 300 MB with WebGPU or on servers, 725 MB on WebAssembly) and caches it. Later calls load from the cache. In the browser the cache is the Cache API. On servers it is `~/.cache/edgewise`, or the folder in `EDGEWISE_CACHE`.

## Stream the output

```ts
const run = generate({ model: 'text:default', input: 'Write a haiku about the sea.' });
for await (const delta of run) process.stdout.write(delta);
const { usage } = await run; // the same Run, awaited, gives the full result
```

## Show download progress

```ts
import { preload } from 'edgewise';

await preload(['text:default', 'voice:default'], {
  onProgress: (e) => {
    if (e.type === 'download') bar.value = e.loaded / e.total;
  },
});
```

Every verb also takes `onProgress`. Events are `download`, `compile` and `ready`.

## Check the device first

```ts
import { capabilities, registry } from 'edgewise';

const caps = await capabilities();
// { runtime: 'browser', webgpu: true, shaderF16: true, tier: 'gpu-high', ... }
const runnable = registry.list({ verb: 'generate', runnable: true });
```

## Bundlers

Edgewise ships ES modules. Vite, webpack, Rspack, esbuild and Bun's bundler all work without plugins. ONNX Runtime loads its `.wasm` files from jsDelivr by default. To self-host them, copy `node_modules/onnxruntime-web/dist/` to your static folder and point Edgewise at it:

```ts
import { configure } from 'edgewise';
configure({ wasmPaths: '/ort/' });
```

## Servers

The same imports work in Bun and Node 20+. On servers, `onnxruntime-node` runs the models on the CPU. See [Runtimes and GPUs](runtimes) for GPU acceleration through vgpu.

```ts file=server.ts
import { writeFileSync } from 'node:fs';
import { speak } from 'edgewise';

const audio = await speak({ model: 'voice:default', input: 'Built on the server.' });
writeFileSync('hello.wav', audio.toWav());
```
