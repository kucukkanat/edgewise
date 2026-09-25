import torch, numpy as np, onnxruntime as ort, time, os, sys
exec(open('router_export.py').read().split("ids,am,tp,cp,pre,post = build")[0].split("w=W(model).eval()")[0])
from onnxruntime.quantization import quantize_dynamic, QuantType
out='/home/claude/exports/lfm2.5-encoder-350m-router'
if not os.path.exists(out+'/onnx/model_quantized.onnx'):
    quantize_dynamic(out+'/onnx/model.onnx', out+'/onnx/model_quantized.onnx', weight_type=QuantType.QInt8)
print('q size', os.path.getsize(out+'/onnx/model_quantized.onnx')/1e6)
exec(open('router_export.py').read().split("ref = model.route")[1].split("ids,am,tp,cp,pre,post = build(prompt, routes)")[0].split("\n",1)[1])
cases=[("Can you help me debug a failing Python unit test?",['Coding','Sales','Creative writing','General knowledge']),
("How much does the enterprise plan cost per seat?",['Coding','Pricing and plans','General conversation']),
("Write me a short poem about the sea in autumn.",['Poetry and creative writing','Tax questions']),
("Ik ben twee keer gefactureerd voor september, kan ik mijn geld terugkrijgen?",['Billing, invoices and refunds','Bugs and outages','Login and account access','Anything else']),
("hi",['Greeting','Complaint','Order status','Technical issue','Feedback','Other'])]
for f in ['model.onnx','model_quantized.onnx']:
    s=ort.InferenceSession(out+'/onnx/'+f); agree=0; md=0
    for text,routes in cases:
        ids,am,tp,cp,_,_ = build(text,routes)
        ref = torch.tensor([ [r['score'] for r in sorted(model.route(text,routes,tokenizer=tok), key=lambda r: routes.index(r['route']))] ])
        t0=time.time(); lg=s.run(None,{'input_ids':ids,'attention_mask':am,'text_pool':tp,'category_pool':cp})[0]; dt=time.time()-t0
        p=torch.tensor(lg).softmax(-1)
        md=max(md,(p-ref).abs().max().item()); agree += int(p.argmax()==ref.argmax())
    print(f,'max prob diff',round(md,4),'argmax agree',agree,'/',len(cases),'last latency ms',round(dt*1000))
