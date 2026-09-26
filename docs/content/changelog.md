---
title: Changelog
group: Reference
order: 40
---

# Changelog

## 0.2.0

- Voice cloning: `speak({ voice: { reference, consent } })`, `cloneVoice()` and saved voices, with Chatterbox Turbo (`voice:clone`). Consent is required.
- Worker mode: `edgewise/worker` runs every verb in a Web Worker with the same API, including streams, tools, schemas and microphone input.
- Models Edgewise exports are all served from the `kucukkanat/edgewise-models` Hugging Face bucket.
- Browser network failures are reported as `DownloadError`.

## 0.1.0

The first release.

- Six verbs: `generate`, `evaluate`, `embed`, `speak`, `paint`, `forecast`, plus `mic()` and `audioSource()` for live audio.
- One isomorphic package for browsers, Bun and Node. WebGPU in browsers, and vgpu for GPU detection and vector maths on servers.
- {{model-count}} models across the verbs, pinned to exact revisions. Chronos-Bolt and the LiquidAI LFM2.5 encoders exported to ONNX by Edgewise, served from a Hugging Face bucket and verified with SHA-256.
- Helpers, React hooks, a Vercel AI SDK provider and mock models for tests.
- Real-model tests on Bun, Node and Chromium.
