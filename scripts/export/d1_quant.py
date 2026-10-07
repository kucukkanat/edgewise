"""Quantize d1-omni's graphs with MatMulNBits (symmetric, block 32), as the other hosted LFM exports.

Usage: python d1_quant.py <onnx-dir> <bits>   -> decide_q{bits}.onnx, vision_q{bits}.onnx, audio_q{bits}.onnx
"""

import os
import sys

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

d, bits = sys.argv[1], int(sys.argv[2])
for name in ("decide", "vision", "audio"):
    q = MatMulNBitsQuantizer(onnx.load(f"{d}/{name}.onnx"), bits=bits, block_size=32, is_symmetric=True, accuracy_level=4)
    q.process()
    path = f"{d}/{name}_q{bits}.onnx"
    q.model.save_model_to_file(path, use_external_data_format=False)
    print("saved", path, round(os.path.getsize(path) / 1e6), "MB")
