import json, sys, numpy as np, onnxruntime as ort
sys.path.insert(0,'scripts'); from router_common import *
from transformers import AutoTokenizer
D='lfm2.5-encoder-350m-prompt-router'
tok=AutoTokenizer.from_pretrained(D)
text="Can you help me debug a failing Python unit test?"; routes=["Coding","Sales","Creative writing"]
ids,tp,cp=pools_offsets(tok,text,routes)
full=prefix(routes)+text
toks=tok.convert_ids_to_tokens(ids.tolist())
s=ort.InferenceSession(D+'/onnx/model.onnx')
feeds={'input_ids':ids[None],'attention_mask':np.ones_like(ids)[None],'text_pool':tp[None],'category_pool':cp[None]}
lg=s.run(None,feeds)[0][0]; pr=np.exp(lg-lg.max()); pr/=pr.sum()
enc=tok(full,return_offsets_mapping=True)
byte_off=[]; pos=0
for i,t in zip(ids.tolist(),toks):
    if i in tok.all_special_ids: byte_off.append([pos,pos])
    else: byte_off.append([pos,pos+len(t)]); pos+=len(t)
doc={
 "model": "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router",
 "task": "zero-shot prompt routing: one encoder pass scores a prompt against N user-defined lanes",
 "license": "LFM Open License v1.0 (license: other / lfm1.0) - see LICENSE",
 "files": {"fp32":"onnx/model.onnx","int8":"onnx/model_quantized.onnx (dynamic int8, per-channel MatMul + int8 embedding Gather)"},
 "inputs": [
   {"name":"input_ids","dtype":"int64","shape":["batch","seq"]},
   {"name":"attention_mask","dtype":"int64","shape":["batch","seq"],"note":"1 for real tokens, 0 for right padding"},
   {"name":"text_pool","dtype":"float32","shape":["batch",1,"seq"],"note":"mean-pool weights over prompt ('Text') tokens: 1/k at each of the k prompt-token positions, 0 elsewhere"},
   {"name":"category_pool","dtype":"float32","shape":["batch","lanes","seq"],"note":"row r = mean-pool weights over the tokens of lane r's description: 1/k_r at each of its k_r token positions, 0 elsewhere"}],
 "outputs": [{"name":"logits","dtype":"float32","shape":["batch","lanes"],"note":"logit[r] = clamp(exp(logit_scale),max=30) * cos(tok_proj(text_rep), rule_proj(lane_rep_r)) + score_bias. Per-lane score = softmax over lanes (as in model.route())."}],
 "build_inputs": {
   "step1_prompt_string": "full = 'Categories:\\n' + lanes.map(l => '- ' + l).join('\\n') + '\\n\\nText:\\n' + prompt   (if lanes is empty the body is '- (none)'; lanes must not contain newlines)",
   "step2_tokenize": "ids = tokenizer.json encode(full) with default post-processor, which PREPENDS <|startoftext|> (id 1). No EOS is appended. No truncation. Transformers.js: `await AutoTokenizer.from_pretrained(dir)` then `tokenizer.encode(full)` (add_special_tokens=true).",
   "step3_token_offsets": "Reference (Python) uses HF offset_mapping in characters. Exact TS-friendly equivalent (verified identical on all tests incl. emoji/CJK): work in UTF-8 BYTE offsets. The tokenizer is byte-level BPE with no normalizer, so each character of a byte-level token string (tokenizer.json model.vocab key, e.g. 'ĠCoding') is exactly one byte of the input. Walk ids in order with pos=0: special/added tokens (ids 0,1,7,16 ...; anything in added_tokens) get offset [pos,pos] (empty); every other token gets [pos, pos+tokenString.length] and pos += tokenString.length. At the end pos must equal utf8Length(full).",
   "step4_text_pool": "textStart = utf8Length(prefix) where prefix = full without the prompt. Prompt tokens = all i with offset.end > textStart and offset.start != offset.end. text_pool[0][0][i] = 1/count for those i.",
   "step5_category_pool": "Lane byte ranges: pos = utf8Length('Categories:\\n'); for each lane r: start = pos + 2 (skip '- '), end = start + utf8Length(lane_r), pos = end + 1 (skip '\\n'). Lane-r tokens = all i with offset.start < end && offset.end > start && offset.start != offset.end (overlap test; note the token ' C' covering the leading space counts). category_pool[0][r][i] = 1/count. If a lane has zero tokens (empty string) its row stays all zero.",
   "step6_mask": "attention_mask = all ones for a single example.",
   "batching": "Right-pad input_ids with 0 (<|pad|>) and attention_mask with 0 to the max seq; pad pools with zeros on the seq axis; pad lanes with all-zero rows and ignore their logits (softmax only over the real lanes of each row). Verified: batched/padded results match per-example results.",
   "postprocess": "probs = softmax(logits[b, 0:N]) ; ranking = sort lanes by prob desc; optional threshold on prob (model.route(threshold=...))."
 },
 "worked_example": {
   "prompt": text, "lanes": routes, "full_string": full,
   "input_ids": ids.tolist(),
   "tokens": toks,
   "char_offsets_hf": [list(o) for o in enc['offset_mapping']],
   "byte_offsets": byte_off,
   "text_start_byte": len(prefix(routes).encode()),
   "lane_byte_ranges": [[s_,e_] for s_,e_ in cat_ranges(routes)],
   "text_token_positions": np.nonzero(tp[0])[0].tolist(),
   "text_pool_value": float(tp[0][np.nonzero(tp[0])[0][0]]),
   "lane_token_positions": [np.nonzero(cp[r])[0].tolist() for r in range(len(routes))],
   "expected_logits_fp32": [round(float(x),5) for x in lg],
   "expected_probs": [round(float(x),5) for x in pr]
 },
 "notes": [
   "logits saturate around +1.1 (match) / -1.64 (no match) because scale*cos+bias is bounded; softmax over lanes therefore never gets extremely peaked (e.g. 0.84 top prob with 4 lanes).",
   "int8 model: top-1 lane agreed with PyTorch on all tests, but non-top lane logits can move by up to ~0.6 on borderline lanes; use fp32 if you rely on calibrated secondary scores.",
   "Transformers.js: use it for the tokenizer only; run the graph with onnxruntime directly (the graph has custom inputs text_pool/category_pool and is not a standard Transformers.js task)."
 ]
}
# lane ranges above are char-based; byte-based for this ASCII example are identical
json.dump(doc,open(D+'/edgewise.json','w'),indent=1,ensure_ascii=False)
print(json.dumps(doc['worked_example'],ensure_ascii=False)[:3000])
