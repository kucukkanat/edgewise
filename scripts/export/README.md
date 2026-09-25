# Model exports

Scripts used to export the models Edgewise hosts itself to ONNX. Paths inside them point at the
working folders used for the original export; adjust them before re-running.

| Script | Model | Output |
| --- | --- | --- |
| `chronos_export.py` | amazon/chronos-bolt-tiny, -small | the `kucukkanat/edgewise-models` bucket |
| `router_export.py`, `router_common.py`, `router_modeling.py`, `q8.py` | LiquidAI LFM2.5-Encoder-350M-Prompt-Router | the `kucukkanat/edgewise-models` bucket |
| `pii_export.py` | LiquidAI LFM2.5-Encoder-350M-PII-Detector | the `kucukkanat/edgewise-models` bucket |
| `*_parity.py`, `*_score.py`, `*_check.py`, `*_diag.py` | | compare ONNX outputs against the PyTorch originals |

8-bit weights use `MatMulNBits` (bits 8, block 32, symmetric). Plain dynamic int8 quantization moved router probabilities by up to 0.35; `MatMulNBits` keeps them within 0.005.
Every hosted file is listed with its SHA-256 in `src/models/manifests.ts`, and Edgewise checks it on download.
