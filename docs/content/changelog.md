---
title: Changelog
group: Reference
order: 40
---

# Changelog

## Unreleased

- `d1-omni-600m` (alias `judge:omni`, preview): LiquidAI's open decision model. It answers `choice()`, `score()` and `boolean()` questions over text, images or speech in one forward pass, with calibrated probabilities. Exported to ONNX by Edgewise (8-bit, checked against the PyTorch original).
- `evaluate({ images, audio, sampleRate })` judges images or an audio clip. `state` is optional when the media are the whole state.

## 0.2.0

- Less memory in WebAssembly browsers: `lfm2.5-350m`, `lfm2.5-230m` and `gemma-3-270m` run 4-bit there from re-exports in the Edgewise bucket (checked by SHA-256). `lfm2.5-350m` drops from about 3.2 GB to 0.7 GB, and `gemma-3-270m` now runs 4-bit on WebAssembly.
- Memory budget: `configure({ memoryBudget })` (default `'auto'`) unloads least recently used models before loading a new one, and on phones throws `OutOfMemoryError` for a model that cannot fit instead of crashing the tab.
- `configure({ preferLowMemory })` (default `'auto'`, on for phones) picks a variant on another device when it needs half the memory or less.
- `memoryUsage()` reports loaded models, their estimated memory and the budget; `estimateMemory()` estimates a variant. Also on the worker API.
- `capabilities()` reports `mobile` (iPads included) and `memory`.
- Voice cloning: `speak({ voice: { reference, consent } })`, `cloneVoice()` and saved voices, with Chatterbox Turbo (`voice:clone`). Consent is required.
- Worker mode: `edgewise/worker` runs every verb in a Web Worker with the same API, including streams, tools, schemas and microphone input.
- Models Edgewise exports are all served from the `kucukkanat/edgewise-models` Hugging Face bucket.
- Browser network failures are reported as `DownloadError`.
- `score()` answers are 0 to 1 as documented (they were the expected level index, 0 to n−1), so score thresholds behave as expected.
- Transcribing less than 0.1 s of audio returns empty text instead of an ONNX Runtime error.
- Barge-in: VAD callbacks `onSpeechStart`, `onSpeechEnd`, `onMisfire` and `onFrame`, plus `mic.speaking` and `mic.speechProbability`.
- `mic({ echoCancellation, noiseSuppression, autoGainControl })` (all default to true).
- `speak().play()` shares one AudioContext and stops at once when the run is cancelled, rejecting with `AbortError`.
- Cancelling is consistent: `cancel()`, an aborted `signal`, and leaving a `for await` loop early all reject the Run with `AbortError`.
- Server downloads: concurrent callers share one download, one caller's abort no longer fails the others, and disk errors reject instead of crashing the process.
- `cache.delete()` matches model repos and never removes saved voices or vector indexes.
- `onnxruntime-web` and `onnxruntime-node` are declared dependencies, pinned to the versions Transformers.js uses.

## 0.1.0

The first release.

- Six verbs: `generate`, `evaluate`, `embed`, `speak`, `paint`, `forecast`, plus `mic()` and `audioSource()` for live audio.
- One isomorphic package for browsers, Bun and Node. WebGPU in browsers, and vgpu for GPU detection and vector maths on servers.
- {{model-count}} models across the verbs, pinned to exact revisions. Chronos-Bolt and the LiquidAI LFM2.5 encoders exported to ONNX by Edgewise, served from a Hugging Face bucket and verified with SHA-256.
- Helpers, React hooks, a Vercel AI SDK provider and mock models for tests.
- Real-model tests on Bun, Node and Chromium.
