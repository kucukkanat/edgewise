---
title: Introduction
group: Get started
order: 1
description: Edgewise runs language, vision, speech, judging, embedding, image and forecasting models on the device, in browsers, Bun and Node, behind six verbs.
---

<div class="hero">
<div>
<p class="eyebrow">edgewise · on-device AI for TypeScript</p>
<h1>Six verbs. Real models. <em>On the device.</em></h1>
<p class="lede">Edgewise runs language, vision, speech, judging, embedding, image and forecasting models where your code runs: in the browser tab through WebGPU or WebAssembly, and in Bun or Node through ONNX Runtime, with <a href="https://github.com/vercel/vgpu">vgpu</a> for WebGPU on the server. You learn six functions, grouped by what they produce.</p>
<div class="ctas"><a class="btn primary" href="quickstart.html">Quickstart</a><div class="install"><span>$</span>bun add edgewise<button class="copy" data-copy="bun add edgewise">Copy</button></div></div>
</div>
<div class="probe" aria-live="polite">
<div class="probe-head">await capabilities()<span class="live"><i></i>your browser</span></div>
<pre id="probeOut"><span class="c">// probing with the real library…</span></pre>
<div class="probe-foot" id="probeFoot">This runs Edgewise's own <code>capabilities()</code> in your browser. Edgewise makes the same check before it downloads anything.</div>
</div>
</div>

## One verb per kind of output

A vision-language model is a language model that also accepts images. A speech recognizer turns audio into text. So both use `generate`. Edgewise has one verb for each kind of output, and every model declares which inputs it accepts.

<div class="verbs">
<a class="verb" href="generate.html"><span class="vo">→ text or object</span><span class="vn">generate</span><span class="vd">Anything that writes text or fills a schema.</span><span class="absorbs"><span class="tchip">tiny LMs</span><span class="tchip">VLMs</span><span class="tchip">tool calling</span><span class="tchip">speech to text</span><span class="tchip">captions</span><span class="tchip">OCR</span></span></a>
<a class="verb" href="evaluate.html"><span class="vo">→ probabilities, spans</span><span class="vn">evaluate</span><span class="vd">Fast decisions on small encoders.</span><span class="absorbs"><span class="tchip">classification</span><span class="tchip">prompt routing</span><span class="tchip">policy lint</span><span class="tchip">PII</span><span class="tchip">injection</span></span></a>
<a class="verb" href="embed.html"><span class="vo">→ vectors</span><span class="vn">embed</span><span class="vd">Meaning as numbers, for search and similarity.</span><span class="absorbs"><span class="tchip">semantic search</span><span class="tchip">RAG</span><span class="tchip">dedupe</span></span></a>
<a class="verb" href="speak.html"><span class="vo">→ audio</span><span class="vn">speak</span><span class="vd">Text, or a live text stream, into speech.</span><span class="absorbs"><span class="tchip">text to speech</span><span class="tchip">voice blending</span><span class="tchip">voice cloning</span></span></a>
<a class="verb" href="paint.html"><span class="vo">→ image</span><span class="vn">paint</span><span class="vd">Images from a prompt, with step previews.</span><span class="absorbs"><span class="tchip">text to image</span></span></a>
<a class="verb" href="forecast.html"><span class="vo">→ numbers with ranges</span><span class="vn">forecast</span><span class="vd">The next values of a time series.</span><span class="absorbs"><span class="tchip">time series</span><span class="tchip">anomaly detection</span></span></a>
</div>

## Verbs compose

Every streaming verb returns a [Run](concepts#runs): `await` it for the result, or iterate it as a stream. A running `generate` can go straight into `speak`. This is a complete local voice assistant for the browser:

```ts file=assistant.ts
import { generate, mic, speak } from 'edgewise';

const microphone = await mic({ vad: true });
for await (const heard of microphone.utterances()) {
  const { text } = await generate({ model: 'stt:realtime', input: heard });   // audio → text
  const reply = generate({ model: 'text:default', input: text });                // text → text, streaming
  await speak({ model: 'voice:default', input: reply }).play();               // text stream → audio
}
```

## The same code in Bun, Node and the browser

Edgewise is one isomorphic TypeScript package. The browser build uses Transformers.js and ONNX Runtime Web (WebGPU, then WebAssembly). The server build uses ONNX Runtime for Node, and probes for a GPU with vgpu. Every result says which one ran:

```ts
const { text, info } = await generate({ model: 'lfm2.5-350m', input: 'Hello!' });
info; // { model: 'lfm2.5-350m', device: 'webgpu', dtype: 'q4f16', backend: 'onnxruntime-web:webgpu' }
```

See [Runtimes and GPUs](runtimes) for what runs where.

## Design principles

1. **Group by output, not by task.** Six verbs cover every task. New model families plug into an existing verb instead of adding API.
2. **Inputs are message parts.** Text, image, audio and video parts work the same everywhere. A model's manifest lists which ones it accepts, and a wrong input fails before any download.
3. **Await or stream, same call.** There are no `streamX` twins. Iterate the Run to stream it.
4. **Honest about devices.** Edgewise picks WebGPU, then WebAssembly or CPU, tells you which one ran, and never hides a fallback.
5. **Pinned, verified models.** Every model is pinned to an exact revision. Files that Edgewise hosts itself are checked against a SHA-256 hash.
6. **Tested for real.** A model is `stable` only when real-model tests pass on Bun, Node and Chromium. See the [model list](models).
