import sys, onnxruntime as ort, torch
from router_common import build, cases
out='/home/claude/exports/lfm2.5-encoder-350m-router/onnx/'
feeds=[build(t,r) for t,r in cases]
base=ort.InferenceSession(out+'model.onnx'); refs=[torch.tensor(base.run(None,f)[0]).softmax(-1) for f in feeds]; del base
for name in sys.argv[1:]:
    s=ort.InferenceSession(out+name); md=0; ag=0
    for f,r in zip(feeds,refs):
        p=torch.tensor(s.run(None,f)[0]).softmax(-1); md=max(md,(p-r).abs().max().item()); ag+=int(p.argmax()==r.argmax())
    print(name, round(md,4), f'{ag}/{len(feeds)}', flush=True)
