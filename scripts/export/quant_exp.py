import sys
from onnxruntime.quantization import quantize_dynamic, QuantType
src,dst,pc,ops,excl=sys.argv[1],sys.argv[2],sys.argv[3]=='1',sys.argv[4].split(','),[x for x in sys.argv[5].split(',') if x]
quantize_dynamic(src,dst,weight_type=QuantType.QInt8,per_channel=pc,op_types_to_quantize=ops,nodes_to_exclude=excl,extra_options={'MatMulConstBOnly':True})
