import torch, numpy as np, onnxruntime as ort, time, sys, os
sys.path.insert(0,'scripts'); from router_common import *
from transformers import AutoModel, AutoTokenizer
src='hf/router'; tok=AutoTokenizer.from_pretrained(src)
model=AutoModel.from_pretrained(src, trust_remote_code=True).eval()
cases=[
 ("Can you help me debug a failing Python unit test?", ["Coding","Sales","Creative writing","General knowledge"]),
 ("Write me a sonnet about autumn leaves falling in Kyoto.", ["Poetry and creative writing","Software engineering","Customer support: refunds and billing"]),
 ("Mi pedido no ha llegado y quiero un reembolso 😡", ["Refund request","Shipping status","Technical issue","Spam"]),
 ("What's the capital of Australia?", ["Math","Geography / trivia"]),
 ("\n  Summarise this contract clause: the lessee shall indemnify ... ", ["Legal document analysis","Medical advice","Translation","Coding","Small talk","Finance – 税金 questions"]),
 ("解一下这个方程：x^2 - 5x + 6 = 0", ["数学","写作","编程"]),
]
paths=sys.argv[1:]
sess={p:ort.InferenceSession(p, providers=['CPUExecutionProvider']) for p in paths}
spec={i for i in tok.all_special_ids}
for text,routes in cases:
    ids,tp,cp=pools_offsets(tok,text,routes)
    tp2,cp2=pools_bytes(src+'/tokenizer.json',ids.tolist(),text,routes,spec)
    same=np.array_equal(tp,tp2) and np.array_equal(cp,cp2)
    feeds={'input_ids':ids[None],'attention_mask':np.ones_like(ids)[None],'text_pool':tp[None],'category_pool':cp[None]}
    with torch.no_grad():
        t=time.time(); ref=model(**{k:torch.from_numpy(v) for k,v in feeds.items()})['logits'].numpy(); tt=time.time()-t
    r_route=model.route(text,routes,tokenizer=tok)
    line=f"S={ids.shape[0]} R={len(routes)} bytepools_match={same} torch={tt*1000:.0f}ms route_top={r_route[0]['route']!r}"
    for p,s in sess.items():
        t=time.time(); o=s.run(None,feeds)[0]; ot=time.time()-t
        pr=lambda x: np.exp(x-x.max())/np.exp(x-x.max()).sum()
        line+=f"\n   {os.path.basename(p)}: maxabs_logit={np.abs(o-ref).max():.2e} maxabs_prob={np.abs(pr(o[0])-pr(ref[0])).max():.2e} argmax_agree={o.argmax()==ref.argmax()} {ot*1000:.0f}ms"
    print(line, '\n   torch logits', np.round(ref[0],3))
# batched with padding
B=[cases[0],cases[2]]
enc=[pools_offsets(tok,t,r) for t,r in B]
S=max(e[0].shape[0] for e in enc); R=max(e[2].shape[0] for e in enc)
ids=np.zeros((2,S),np.int64); am=np.zeros((2,S),np.int64); tp=np.zeros((2,1,S),np.float32); cp=np.zeros((2,R,S),np.float32)
for i,(a,b,c) in enumerate(enc):
    n=a.shape[0]; ids[i,:n]=a; am[i,:n]=1; tp[i,:,:n]=b; cp[i,:c.shape[0],:n]=c
feeds={'input_ids':ids,'attention_mask':am,'text_pool':tp,'category_pool':cp}
with torch.no_grad(): ref=model(**{k:torch.from_numpy(v) for k,v in feeds.items()})['logits'].numpy()
for p,s in sess.items():
    o=s.run(None,feeds)[0]; print('batched-padded',os.path.basename(p),'maxabs (valid lanes)',max(np.abs(o[0]-ref[0]).max(),np.abs(o[1,:4]-ref[1,:4]).max()))
# padded vs unpadded equivalence in torch
single=model(**{k:torch.from_numpy(v[:1,:enc[0][0].shape[0]]) if k!='category_pool' else torch.from_numpy(v[:1,:3,:enc[0][0].shape[0]]) for k,v in feeds.items()}) if False else None
