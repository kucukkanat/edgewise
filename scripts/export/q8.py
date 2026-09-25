import onnx, os, sys, inspect
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
print(inspect.signature(MatMulNBitsQuantizer.__init__))
d=sys.argv[1]
m=onnx.load(d+'model.onnx'); q=MatMulNBitsQuantizer(m, bits=8, block_size=32, is_symmetric=True, accuracy_level=4); q.process(); q.model.save_model_to_file(d+'model_q8.onnx', use_external_data_format=False)
print('saved', os.path.getsize(d+'model_q8.onnx')/1e6)
