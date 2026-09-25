import torch, sys, time
from transformers import AutoModel, AutoTokenizer
torch.manual_seed(0)
src='hf/router'
tok=AutoTokenizer.from_pretrained(src)
model=AutoModel.from_pretrained(src, trust_remote_code=True, attn_implementation=sys.argv[2] if len(sys.argv)>2 else 'eager').eval()
print(model.config._attn_implementation)
print(model.route("Can you help me debug a failing Python unit test?", ["Coding","Sales","Creative writing","General knowledge"], tokenizer=tok))

class W(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,input_ids,attention_mask,text_pool,category_pool):
        return s.m(input_ids=input_ids,attention_mask=attention_mask,text_pool=text_pool,category_pool=category_pool)["logits"]
w=W(model).eval()
B,S,R=2,24,3
ids=torch.randint(10,1000,(B,S)); am=torch.ones(B,S,dtype=torch.long); am[1,-5:]=0
tp=torch.rand(B,1,S); cp=torch.rand(B,R,S)
dyn = sys.argv[1]=='dynamo'
out='onnx_tmp/router.onnx'
import os; os.makedirs('onnx_tmp',exist_ok=True)
t=time.time()
if dyn:
    from torch.export import Dim
    b=Dim('batch'); s=Dim('seq',min=2,max=32768); r=Dim('lanes',min=1, max=1024)
    torch.onnx.export(w,(ids,am,tp,cp),out,dynamo=True,opset_version=18,
        input_names=['input_ids','attention_mask','text_pool','category_pool'],output_names=['logits'],
        dynamic_shapes={'input_ids':{0:b,1:s},'attention_mask':{0:b,1:s},'text_pool':{0:b,2:s},'category_pool':{0:b,1:r,2:s}}, external_data=True)
else:
    with torch.no_grad():
        torch.onnx.export(w,(ids,am,tp,cp),out,dynamo=False,opset_version=17,
        input_names=['input_ids','attention_mask','text_pool','category_pool'],output_names=['logits'],
        dynamic_axes={'input_ids':{0:'batch',1:'seq'},'attention_mask':{0:'batch',1:'seq'},'text_pool':{0:'batch',2:'seq'},'category_pool':{0:'batch',1:'lanes',2:'seq'},'logits':{0:'batch',1:'lanes'}})
print('export time',time.time()-t)
