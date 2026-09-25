import torch, numpy as np, onnxruntime as ort
from transformers import AutoModelForTokenClassification, AutoTokenizer
mid='LiquidAI/LFM2.5-Encoder-350M-PII-Detector'; rev='b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2'
tok=AutoTokenizer.from_pretrained(mid, revision=rev, trust_remote_code=True)
model=AutoModelForTokenClassification.from_pretrained(mid, revision=rev, trust_remote_code=True, attn_implementation='eager').eval()
s=ort.InferenceSession('/home/claude/exports/lfm2.5-encoder-350m-pii/onnx/model.onnx')
for t in ["Email Dr. Laura Schmidt at laura@charite.de or call +49 30 1234567.","Mijn IBAN is NL91ABNA0417164300, stuur het naar Kerkstraat 12, Almere.","hi","My password is hunter2 and my card is 4111 1111 1111 1111."]:
    e=tok(t, return_tensors='pt')
    with torch.no_grad(): ref=model(**e).logits.numpy()
    o=s.run(None,{'input_ids':e['input_ids'].numpy(),'attention_mask':e['attention_mask'].numpy()})[0]
    print(e['input_ids'].shape[1], np.abs(o-ref).max(), (o.argmax(-1)!=ref.argmax(-1)).sum())
