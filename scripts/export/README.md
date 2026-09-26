# Model exports

Scripts used to export the models Edgewise hosts itself to ONNX. Paths inside them point at the
working folders used for the original export; adjust them before re-running.

| Script | Model | Output |
| --- | --- | --- |
| `chronos_export.py` | amazon/chronos-bolt-tiny, -small | the `kucukkanat/edgewise-models` bucket |
| `router_export.py`, `router_common.py`, `router_modeling.py`, `q8.py` | LiquidAI LFM2.5-Encoder-350M-Prompt-Router | the `kucukkanat/edgewise-models` bucket |
| `pii_export.py` | LiquidAI LFM2.5-Encoder-350M-PII-Detector | the `kucukkanat/edgewise-models` bucket |
| `wasm_q4.py`, `wasm_gather.py` | LFM2.5-350M, LFM2.5-230M, Gemma 3 270M (4-bit) | the `kucukkanat/edgewise-models` bucket, `<model>/wasm-q4/v1/` |
| `*_parity.py`, `*_score.py`, `*_check.py`, `*_diag.py` | | compare ONNX outputs against the PyTorch originals |

8-bit weights use `MatMulNBits` (bits 8, block 32, symmetric). Plain dynamic int8 quantization moved router probabilities by up to 0.35; `MatMulNBits` keeps them within 0.005.
Every hosted file is listed with its SHA-256 in `src/models/manifests.ts`, and Edgewise checks it on download.

`wasm_q4.py` makes 4-bit exports run on ONNX Runtime Web's WebAssembly build, which has no `GatherBlockQuantized` kernel. `wasm_gather.py` replaces each `GatherBlockQuantized` with standard operators: gather the packed rows, unpack the nibbles with `BitShift`, subtract the zero points and multiply by the scales. The opset is unchanged and outputs match the original bit for bit (the script checks on native ONNX Runtime). Usage: `python wasm_q4.py <repo> <revision> <out-dir> onnx/model_q4.onnx`.
