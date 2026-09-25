import json, sys, os, time, torch, numpy as np
from chronos import BaseChronosPipeline
name = sys.argv[1]  # e.g. amazon/chronos-bolt-tiny
rev = sys.argv[2]
out = sys.argv[3]
pipe = BaseChronosPipeline.from_pretrained(name, revision=rev, device_map='cpu', torch_dtype=torch.float32)
model = pipe.model.eval()
cc = model.chronos_config
print('config', cc)

class Core(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, patched_context, attention_mask):
        m = self.m
        input_embeds = m.input_patch_embedding(patched_context)
        am = attention_mask
        if m.chronos_config.use_reg_token:
            b = input_embeds.shape[0]
            reg = m.shared(torch.full((b, 1), m.config.reg_token_id, dtype=torch.long))
            input_embeds = torch.cat([input_embeds, reg], dim=-2)
            am = torch.cat([am, torch.ones((b, 1), dtype=am.dtype)], dim=-1)
        hidden = m.encoder(attention_mask=am, inputs_embeds=input_embeds)[0]
        b = input_embeds.shape[0]
        dec_ids = torch.zeros((b, 1), dtype=torch.long) + int(m.config.decoder_start_token_id)
        seq = m.decoder(input_ids=dec_ids, encoder_hidden_states=hidden, encoder_attention_mask=am, return_dict=True).last_hidden_state
        q = m.output_patch_embedding(seq).view(input_embeds.shape[0], m.num_quantiles, m.chronos_config.prediction_length)
        return q

core = Core(model).eval()
P = cc.input_patch_size
def prep(ctx):
    x = torch.tensor(ctx, dtype=torch.float32)[None]
    x = x[..., -cc.context_length:]
    mask = (~torch.isnan(x)).float()
    loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num((x-loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
    scale = torch.where(scale == 0, 1e-5, scale)
    xs = (x-loc)/scale
    L = xs.shape[-1]
    if L % P:
        pad = P - L % P
        xs = torch.cat([torch.full((1,pad), float('nan')), xs], -1); mask = torch.cat([torch.full((1,pad), float('nan')), mask], -1)
    pc = xs.unfold(-1, P, cc.input_patch_stride); pm = torch.nan_to_num(mask.unfold(-1, P, cc.input_patch_stride), nan=0.0)
    pc = torch.where(pm > 0, pc, 0.0)
    return torch.cat([pc, pm], -1), (pm.sum(-1) > 0).float(), loc, scale

rng = np.random.default_rng(0)
t = np.arange(300)
series = [np.sin(t/7)*10+50, rng.normal(0,1,300).cumsum(), np.r_[np.full(100, np.nan), np.cos(t[:200]/5)*3], np.arange(40)*2.0, rng.poisson(5,500).astype(float)]
pc, am, loc, scale = prep(series[0])
os.makedirs(out+'/onnx', exist_ok=True)
torch.onnx.export(core, (pc, am), out+'/onnx/model.onnx', input_names=['patched_context','attention_mask'], output_names=['quantiles'],
  dynamic_axes={'patched_context':{0:'batch',1:'patches'}, 'attention_mask':{0:'batch',1:'patches'}, 'quantiles':{0:'batch'}}, opset_version=17, dynamo=False)
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(out+'/onnx/model.onnx', out+'/onnx/model_quantized.onnx', weight_type=QuantType.QInt8)
for f in ['model.onnx','model_quantized.onnx']:
    s = ort.InferenceSession(out+'/onnx/'+f)
    md=0
    for ser in series:
        pc, am, loc, scale = prep(ser)
        with torch.no_grad():
            ref = pipe.predict_quantiles(torch.tensor(ser, dtype=torch.float32)[None], prediction_length=cc.prediction_length)[0][0].numpy()  # [H, Q]
        t0=time.time(); q = s.run(None, {'patched_context': pc.numpy(), 'attention_mask': am.numpy()})[0][0]; dt=time.time()-t0
        qq = q * scale.numpy()[0] + loc.numpy()[0]  # [Q,H]
        d = np.abs(qq.T - ref).max() / (np.abs(ref).max()+1e-9)
        md=max(md,d)
    print(f, 'max rel diff vs pytorch pipeline', md, 'latency', round(dt*1000,1),'ms', os.path.getsize(out+'/onnx/'+f)/1e6,'MB')
meta = {'quantiles': [float(x) for x in model.chronos_config.quantiles], 'prediction_length': cc.prediction_length, 'context_length': cc.context_length,
  'patch_size': P, 'patch_stride': cc.input_patch_stride, 'use_arcsinh': bool(getattr(cc,'use_arcsinh',False)), 'use_reg_token': bool(cc.use_reg_token),
  'inputs': {'patched_context': '[batch, patches, 2*patch_size] float32: normalized values then 0/1 observed mask', 'attention_mask': '[batch, patches] float32'},
  'outputs': {'quantiles': '[batch, num_quantiles, prediction_length] float32, normalized; multiply by scale and add loc'},
  'source': {'repo': name, 'revision': rev}}
json.dump(meta, open(out+'/edgewise.json','w'), indent=1)
print(json.dumps(meta))
