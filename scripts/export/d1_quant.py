"""Quantize d1 graphs with MatMulNBits (symmetric, block 32), as the other hosted LFM exports.

Usage: python d1_quant.py <onnx-dir> <bits> [name,name,...] [--gather]
  names default to decide,vision,audio (d1-omni-600M); d1-3B's are embed,lower,upper,vision.
  --gather also quantizes Gather tables (d1-3B's 128k-token embedding) to GatherBlockQuantized, which ONNX Runtime
  runs on Node, Bun and WebGPU but not on its WebAssembly build.
Writes <name>_q<bits>.onnx as single files (each must stay under 2 GB).
"""

import os
import sys

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

args = [a for a in sys.argv[1:] if not a.startswith("--")]
d, bits = args[0], int(args[1])
names = args[2].split(",") if len(args) > 2 else ["decide", "vision", "audio"]
gather = "--gather" in sys.argv
for name in names:
    extra = {"op_types_to_quantize": ("MatMul", "Gather"), "quant_axes": (("MatMul", 0), ("Gather", 1))} if gather else {}
    q = MatMulNBitsQuantizer(onnx.load(f"{d}/{name}.onnx"), bits=bits, block_size=32, is_symmetric=True,
                             accuracy_level=4, **extra)
    q.process()
    path = f"{d}/{name}_q{bits}.onnx"
    q.model.save_model_to_file(path, use_external_data_format=False)
    print("saved", path, round(os.path.getsize(path) / 1e6), "MB")
