---
title: Roadmap
group: Reference
order: 50
---

# Roadmap

Things we want to add, roughly in order. None of them are in the package yet.

## Voice cloning

`speak({ voice: { reference, consent } })` is reserved. It needs a cloning model that runs well on the device and has a licence that allows it. Candidates are being evaluated. Consent will be required in the API: `consent: { attested: true }`.

## Worker mode

Run models in a dedicated worker in the browser so the page never blocks. Today, heavy work runs on the calling thread; ONNX Runtime's WebAssembly backend already uses its own threads when the page is cross-origin isolated.

## More models

- Image embeddings (SigLIP) for `embed` with image input.
- Larger VLMs and text models as WebGPU memory limits rise.
- Faster TTS for low-end devices.
- More paint models once one-step models fit in browser memory.

## Grammar-constrained decoding

`schema` currently validates and retries. Constrained decoding would guarantee valid JSON on the first try.
