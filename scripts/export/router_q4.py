import onnx, os
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
import sys; out=sys.argv[1] if len(sys.argv)>1 else '/home/claude/exports/lfm2.5-encoder-350m-router/onnx/'
m=onnx.load(out+'model.onnx'); q=MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, accuracy_level=4); q.process(); q.model.save_model_to_file(out+'model_q4.onnx', use_external_data_format=False)
print('saved', os.path.getsize(out+'model_q4.onnx')/1e6)
