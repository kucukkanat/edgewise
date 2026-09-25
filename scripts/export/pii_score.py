import sys, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('LiquidAI/LFM2.5-Encoder-350M-PII-Detector', revision='b8c9cf3d2d6ae52501b35a27ba46f271449c9ce2', trust_remote_code=True)
d='/home/claude/exports/lfm2.5-encoder-350m-pii/onnx/'
texts=["Email Dr. Laura Schmidt at laura@charite.de or call +49 30 1234567.","Mijn IBAN is NL91ABNA0417164300, stuur het naar Kerkstraat 12, Almere.","No personal data here, just a normal sentence about weather.","My password is hunter2 and my card is 4111 1111 1111 1111.","Contact John Smith (john.smith@example.com), SSN 123-45-6789, born 1984-03-12.","Server at 192.168.1.20, api key sk-live-9f8a7b6c5d4e3f2a1b0c."]
a=ort.InferenceSession(d+'model.onnx'); b=ort.InferenceSession(d+''+(sys.argv[1] if len(sys.argv)>1 else 'model_q4.onnx')); ag=0; tot=0; spanag=0
for t in texts:
    e=tok(t, return_tensors='np'); f={'input_ids':e['input_ids'].astype(np.int64),'attention_mask':e['attention_mask'].astype(np.int64)}
    x=a.run(None,f)[0].argmax(-1); y=b.run(None,f)[0].argmax(-1); ag+=(x==y).sum(); tot+=x.size
print('q4 token agree', ag, '/', tot)
