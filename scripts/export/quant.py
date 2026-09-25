import sys, os
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
src,dst=sys.argv[1],sys.argv[2]
ops=sys.argv[3].split(',') if len(sys.argv)>3 else ['MatMul','Gather']
quantize_dynamic(src,dst,weight_type=QuantType.QInt8,per_channel=os.environ.get("PC")=="1",reduce_range=False,op_types_to_quantize=ops,
                 extra_options={'MatMulConstBOnly':True})
print(dst, os.path.getsize(dst)/1e6,'MB')
