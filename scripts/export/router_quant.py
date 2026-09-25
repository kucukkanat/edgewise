import numpy as np, onnxruntime as ort, os, onnx, torch
from router_common import build, cases
from onnxruntime.quantization import quantize_dynamic, QuantType
out='/home/claude/exports/lfm2.5-encoder-350m-router/onnx/'
feeds=[build(t,r) for t,r in cases]
base=ort.InferenceSession(out+'model.onnx'); refs=[torch.tensor(base.run(None,f)[0]).softmax(-1) for f in feeds]; del base
def score(path):
    s=ort.InferenceSession(path); md=0; ag=0
    for f,r in zip(feeds,refs):
        p=torch.tensor(s.run(None,f)[0]).softmax(-1); md=max(md,(p-r).abs().max().item()); ag+=int(p.argmax()==r.argmax())
    return round(md,4), ag
print('int8 default', score(out+'model_quantized.onnx'), flush=True)
if not os.path.exists(out+'model_int8pc.onnx'):
    quantize_dynamic(out+'model.onnx', out+'model_int8pc.onnx', weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=['MatMul'])
print('int8 per-channel matmul-only', score(out+'model_int8pc.onnx'), os.path.getsize(out+'model_int8pc.onnx')/1e6, flush=True)
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
if not os.path.exists(out+'model_q4.onnx'):
    m=onnx.load(out+'model.onnx'); q=MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, accuracy_level=4); q.process(); q.model.save_model_to_file(out+'model_q4.onnx', use_external_data_format=False)
print('q4', score(out+'model_q4.onnx'), os.path.getsize(out+'model_q4.onnx')/1e6, flush=True)
