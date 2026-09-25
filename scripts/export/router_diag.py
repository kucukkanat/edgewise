import torch, numpy as np, onnxruntime as ort
from transformers import AutoModel
from router_common import build, cases, mid, rev
model = AutoModel.from_pretrained(mid, revision=rev, trust_remote_code=True, attn_implementation='eager').eval()
s=ort.InferenceSession('/home/claude/exports/lfm2.5-encoder-350m-router/onnx/model.onnx')
for t,r in cases:
    f=build(t,r)
    with torch.no_grad(): ref=model(**{k:torch.tensor(v) for k,v in f.items()})['logits'].numpy()
    o=s.run(None,f)[0]; print(f['input_ids'].shape[1], len(r), np.abs(o-ref).max())
