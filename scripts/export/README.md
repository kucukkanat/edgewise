# Model exports

Scripts used to export the models Edgewise hosts itself to ONNX. Paths inside them point at the
working folders used for the original export; adjust them before re-running.

| Script | Model | Output |
| --- | --- | --- |
| `chronos_export.py` | amazon/chronos-bolt-tiny, -small | the `kucukkanat/edgewise-models` bucket |
| `router_export.py`, `router_common.py`, `router_modeling.py`, `q8.py` | LiquidAI LFM2.5-Encoder-350M-Prompt-Router | the `kucukkanat/edgewise-models` bucket |
| `pii_export.py` | LiquidAI LFM2.5-Encoder-350M-PII-Detector | the `kucukkanat/edgewise-models` bucket |
| `d1_export.py`, `d1_quant.py`, `d1_parity.py` | LiquidAI d1-omni-600M (decision, vision and audio graphs) | the `kucukkanat/edgewise-models` bucket, `d1-omni-600m/v1/` |
| `d1_3b_export.py`, `d1_quant.py`, `d1_3b_parity.py` | LiquidAI d1-3B (embedding, two decoder halves, vision) | the `kucukkanat/edgewise-models` bucket, `d1-3b/v1/` |
| `wasm_q4.py`, `wasm_gather.py` | LFM2.5-350M, LFM2.5-230M, Gemma 3 270M (4-bit) | the `kucukkanat/edgewise-models` bucket, `<model>/wasm-q4/v1/` |
| `*_parity.py`, `*_score.py`, `*_check.py`, `*_diag.py` | | compare ONNX outputs against the PyTorch originals |

8-bit weights use `MatMulNBits` (bits 8, block 32, symmetric). Plain dynamic int8 quantization moved router probabilities by up to 0.35; `MatMulNBits` keeps them within 0.005.
Every hosted file is listed with its SHA-256 in `src/models/manifests.ts`, and Edgewise checks it on download.

`wasm_q4.py` makes 4-bit exports run on ONNX Runtime Web's WebAssembly build, which has no `GatherBlockQuantized` kernel. `wasm_gather.py` replaces each `GatherBlockQuantized` with standard operators: gather the packed rows, unpack the nibbles with `BitShift`, subtract the zero points and multiply by the scales. The opset is unchanged and outputs match the original bit for bit (the script checks on native ONNX Runtime). Usage: `python wasm_q4.py <repo> <revision> <out-dir> onnx/model_q4.onnx`.

d1-omni-600M: `python d1_export.py <hf-dir> <out>` writes `decide.onnx` (trunk and decision head), `vision.onnx` (one image crop per run) and `audio.onnx`; `python d1_quant.py <out> 8` quantizes them; `python d1_parity.py <hf-dir> <out> "" _q8` checks them against `model.probabilities()` on text, JSON, small and tiled images and two audio lengths. fp32 matches PyTorch exactly; 8-bit keeps every top answer, with probabilities within 0.038 (worst case, batched). 4-bit changed a top answer and moved a probability by 0.4, so it is not hosted. The vision tower is fine-tuned (it differs from LFM2.5-VL-450M's), so LiquidAI's VL ONNX cannot be reused. Two rewrites keep the graphs portable: SigLIP2's antialiased position-embedding resize is passed in as weight matrices, because ONNX Runtime's WebGPU provider has no antialiased `Resize`, and the conformer's relative-position shift is a gather, so the audio length stays dynamic.

d1-3B: `python d1_3b_export.py <hf-dir> <out>` exports `embed`, `lower` (decoder layers 0-14), `upper` (layers 15-29, final norm and head at the last position) and `vision` in fp32 with external data, after checking its plain forward against HF's logits (identical). Splitting the decoder keeps each 8-bit file under 2 GB, so Edgewise loads them without external data. `python d1_quant.py <out> 8 lower,upper,vision` quantizes; the embedding table stays full precision (`embed_q8.onnx` is published as `embed.onnx`), because ONNX Runtime quantizes Gather only at 4 bits, and a 4-bit table moved probabilities by up to 0.046 against 0.004 for 8-bit everywhere else. `python d1_3b_parity.py <hf-dir> <out> "" _q8` compares with d1-3B's own `probabilities()` on text, JSON, a 12-way choice, a small and a tiled image: fp32 exact, 8-bit within 0.0043, every top answer kept. ONNX Runtime Web cannot allocate the 1.4 GB decoder halves in WebAssembly memory, so the model is marked `requires: { server: true }`.
