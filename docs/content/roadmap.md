---
title: Roadmap
group: Reference
order: 50
---

# Roadmap

Things we want to add, roughly in order. None of them are in the package yet.

## Voice cloning without WebGPU

Chatterbox's quantized graphs use an ONNX operator the WebAssembly build of ONNX Runtime lacks. A WebAssembly-friendly export would let cloning run in browsers without a GPU. Multilingual cloning is also on the list.


## More models

- Image embeddings (SigLIP) for `embed` with image input.
- Larger VLMs and text models as WebGPU memory limits rise.
- Faster TTS for low-end devices.
- More paint models once one-step models fit in browser memory.

## Grammar-constrained decoding

`schema` currently validates and retries. Constrained decoding would guarantee valid JSON on the first try.
