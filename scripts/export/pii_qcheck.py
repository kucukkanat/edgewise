import numpy as np, onnxruntime as ort, sys, gc
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('hf/pii')
texts=open('scripts/pii_texts.txt').read().split('\n')
ref=ort.InferenceSession('lfm2.5-encoder-350m-pii-detector/onnx/model.onnx')
R=[ref.run(None,{'input_ids':np.array([tok(t)['input_ids']]),'attention_mask':np.ones((1,len(tok(t)['input_ids'])),np.int64)})[0] for t in texts]
del ref; gc.collect()
for p in sys.argv[1:]:
    s=ort.InferenceSession(p); agree=[]; mx=0
    for t,r in zip(texts,R):
        ids=np.array([tok(t)['input_ids']]); o=s.run(None,{'input_ids':ids,'attention_mask':np.ones_like(ids)})[0]
        agree.append((o.argmax(-1)==r.argmax(-1)).mean()); mx=max(mx,np.abs(o-r).max())
    print(p, 'agree',np.round(agree,3),'mean',np.mean(agree).round(4),'maxabs',mx)
    del s; gc.collect()
