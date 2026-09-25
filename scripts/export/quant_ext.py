import sys, onnx, os
from onnxruntime.quantization import quantize_dynamic, QuantType
src,dst,pc,ops,excl=sys.argv[1],sys.argv[2],sys.argv[3]=='1',sys.argv[4].split(','),[x for x in sys.argv[5].split(',') if x]
tmp=dst+'.ext.onnx'
quantize_dynamic(src,tmp,weight_type=QuantType.QInt8,per_channel=pc,op_types_to_quantize=ops,nodes_to_exclude=excl,use_external_data_format=True,extra_options={'MatMulConstBOnly':True})
m=onnx.load(tmp); onnx.save(m,dst)
d=os.path.dirname(tmp) or '.'
os.remove(tmp)
for f in os.listdir(d):
    if f.startswith(os.path.basename(tmp)) and f!=os.path.basename(dst): os.remove(os.path.join(d,f))
print(dst, os.path.getsize(dst)/1e6)
