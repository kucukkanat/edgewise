import torch, numpy as np, onnxruntime as ort, time, sys, os, types
sys.path.insert(0,'hf/pii'); import pii_hybrid_decode as hd
from transformers import AutoModelForTokenClassification, AutoTokenizer
src='hf/pii'; tok=AutoTokenizer.from_pretrained(src)
model=AutoModelForTokenClassification.from_pretrained(src, trust_remote_code=True).eval()
texts=["Email Dr. Laura Schmidt at laura@charite.de.",
 "My name is John Smith, SSN 123-45-6789, and I live at 42 Wallaby Way, Sydney NSW 2000. Call me on +61 2 9374 4000.",
 "Herr Müller (geb. 12.03.1985) hat Diabetes und nimmt Metformin 500mg. IBAN DE89 3704 0044 0532 0130 00.",
 "我叫张敏，电话是13812345678，住在北京市朝阳区建国路88号。",
 "postgres://admin:S3cr3tPass@db.internal:5432/prod  api key sk-proj-abcdefghijklmnopqrstuvwx1234 from 192.168.1.20",
 "Nothing sensitive here, just a sentence about the weather in spring.",
 "Patient MRN: A1234567, insured by Aetna, policy # HP-99812. Visit https://portal.example.com/u/jdoe as user jdoe_88."]
paths=sys.argv[1:]; sess={p:ort.InferenceSession(p) for p in paths}
class OrtModel:
    def __init__(s,se): s.s=se; s.device='cpu'; s.config=model.config
    def __call__(s,input_ids,attention_mask,**k):
        return types.SimpleNamespace(logits=torch.from_numpy(s.s.run(None,{'input_ids':input_ids.numpy(),'attention_mask':attention_mask.numpy()})[0]))
for t in texts:
    enc=tok(t,return_tensors='np'); feeds={'input_ids':enc['input_ids'].astype(np.int64),'attention_mask':enc['attention_mask'].astype(np.int64)}
    with torch.no_grad():
        t0=time.time(); ref=model(**{k:torch.from_numpy(v) for k,v in feeds.items()}).logits.numpy(); tt=time.time()-t0
    rs=hd.predict(t,tok,model,hybrid=False); rh=hd.predict(t,tok,model)
    print(f"S={ref.shape[1]} torch {tt*1000:.0f}ms spans={[(s['type'],s['text']) for s in rs]}")
    for p,s in sess.items():
        t0=time.time(); o=s.run(None,feeds)[0]; ot=time.time()-t0
        om=OrtModel(s)
        print(f"   {os.path.basename(p)}: maxabs={np.abs(o-ref).max():.2e} token_argmax_agree={(o.argmax(-1)==ref.argmax(-1)).mean()*100:.1f}% ({(o.argmax(-1)!=ref.argmax(-1)).sum()} diff) raw_spans_equal={hd.predict(t,tok,om,hybrid=False)==rs} hybrid_spans_equal={hd.predict(t,tok,om)==rh} {ot*1000:.0f}ms")
# batched padded
enc=tok(texts[:3],padding=True,return_tensors='np'); feeds={'input_ids':enc['input_ids'].astype(np.int64),'attention_mask':enc['attention_mask'].astype(np.int64)}
print('padding side',tok.padding_side, feeds['attention_mask'].sum(1))
with torch.no_grad(): ref=model(**{k:torch.from_numpy(v) for k,v in feeds.items()}).logits.numpy()
for p,s in sess.items():
    o=s.run(None,feeds)[0]; m=feeds['attention_mask'].astype(bool); print('batched',os.path.basename(p),np.abs(o-ref)[m].max(), (o.argmax(-1)==ref.argmax(-1))[m].mean())
