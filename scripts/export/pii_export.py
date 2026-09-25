import torch, time, os
from transformers import AutoModelForTokenClassification
src='hf/pii'
model=AutoModelForTokenClassification.from_pretrained(src, trust_remote_code=True).eval()
class W(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,input_ids,attention_mask): return s.m(input_ids=input_ids,attention_mask=attention_mask).logits
w=W(model).eval()
ids=torch.randint(10,1000,(2,24)); am=torch.ones(2,24,dtype=torch.long); am[1,-5:]=0
D='lfm2.5-encoder-350m-pii-detector/onnx'; os.makedirs(D,exist_ok=True)
t=time.time()
with torch.no_grad():
    torch.onnx.export(w,(ids,am),D+'/model.onnx',dynamo=False,opset_version=17,
      input_names=['input_ids','attention_mask'],output_names=['logits'],
      dynamic_axes={'input_ids':{0:'batch',1:'seq'},'attention_mask':{0:'batch',1:'seq'},'logits':{0:'batch',1:'seq'}})
print('export',time.time()-t)
