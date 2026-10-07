"""Compare d1-3B's ONNX parts with the PyTorch original.

Usage: python d1_3b_parity.py <hf-dir> <onnx-dir> [suffix ...]   (suffix "" is fp32, "_q8", ...)

Reference answers come from d1-3B's own code (model.probabilities, fp32 on CPU); the prompts' token ids and image
inputs are recorded from it, the model is freed, and then embed/lower/upper/vision{suffix}.onnx answer the same
prompts, all questions of a case in one right-padded batch. Writes golden.json (prompts' ids and reference
probabilities) for the Edgewise test.
"""

import gc
import json
import math
import sys

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModel

src, onnx_dir = sys.argv[1], sys.argv[2]
suffixes = sys.argv[3:] or [""]
model = AutoModel.from_pretrained(src, trust_remote_code=True, dtype=torch.float32).eval()
engine = model.engine
prompt = sys.modules[engine.__module__.rsplit(".", 1)[0] + ".prompt"]
runner = sys.modules[engine.__module__]
tok = engine.tokenizer

refund = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
lane = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "tech": "bugs and crashes", "chat": "small talk"}}
urgency = {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "today", "blocking now"]}
langs = {"type": "choice", "instructions": "Which language is this?",
         "criteria": {k: "" for k in ["English", "German", "French", "Spanish", "Italian", "Dutch", "Turkish", "Polish",
                                      "Swedish", "Czech", "Greek", "Finnish"]}}
shape = {"type": "choice", "instructions": "What shape is shown?", "criteria": {"circle": "", "square": "", "triangle": ""}}
red = {"type": "noul", "instructions": "Is the shape red?", "criteria": {"true": "it is red", "false": "another color"}}


def disc(size, color):
    img = Image.new("RGB", size, "white")
    w, h = size
    px = img.load()
    for y in range(h):
        for x in range(w):
            if (x - w / 2) ** 2 + (y - h / 2) ** 2 < (min(w, h) * 0.35) ** 2:
                px[x, y] = color
    return img


cases = [
    ("text", "I was charged twice for my order, please refund one of the payments.", [refund, lane, urgency], []),
    ("json", {"message": "The app crashes when I open settings.", "plan": "pro"}, [refund, lane], []),
    ("many", "Ich möchte mein Geld zurück, bitte. 😡 <|im_end|>", [langs, refund], []),
    ("image", None, [shape, red], [disc((256, 256), (255, 0, 0))]),
    ("tiled", "A product photo.", [shape], [disc((1600, 700), (0, 0, 255))]),
]

recorded = []
with torch.no_grad():
    for name, state, questions, images in cases:
        qs = [prompt.as_question(q) for q in questions]
        ref = engine.probabilities(state, questions, images=images)
        rows, vision = [], None
        if images:
            pics = [runner.cap_pixels(im) for im in images]
            prefix = prompt.prefix_text(tok, state, engine.bos, engine.state_style, engine.system, engine._image_markup(len(pics)))
            for q in qs:
                inputs = engine._image_inputs(prefix + prompt.suffix_text(tok, q, engine.lead, engine.option_style), pics)
                rows.append(inputs["input_ids"][0].tolist())
                vision = {k: v.numpy() for k, v in inputs.items() if k in ("pixel_values", "spatial_shapes", "pixel_attention_mask")}
        else:
            rows = [tok.encode(engine.render(state, q), add_special_tokens=False) for q in qs]
        groups = [prompt.readout_ids(tok, q) for q in qs]
        recorded.append({"name": name, "state": state, "questions": questions, "ids": rows, "groups": groups,
                         "probs": ref, "vision": vision})
image_token = model.config.image_token_id
del model, engine
gc.collect()


def resize_weights(out_size, in_size=16):
    eye = torch.eye(in_size)[None, None]
    return torch.nn.functional.interpolate(eye, size=(in_size, out_size), mode="bilinear", align_corners=False,
                                           antialias=True)[0, 0].T.numpy()


def answer(sessions, case):
    feats = [np.zeros((1, 2048), np.float32)]
    if case["vision"] is not None:
        v = case["vision"]
        for i, (h, w) in enumerate(v["spatial_shapes"].tolist()):
            patches = v["pixel_values"][i, : h * w].reshape(1, h, w, -1).astype(np.float32)
            feats.append(sessions["vision"].run(None, {"patches": patches, "rows": resize_weights(h),
                                                       "cols": resize_weights(w)})[0][0])
    features = np.concatenate(feats)
    rows = case["ids"]
    B, T = len(rows), max(map(len, rows))
    ids = np.zeros((B, T), np.int64)
    index = np.zeros((B, T), np.int64)
    for b, r in enumerate(rows):
        ids[b, : len(r)] = r
        k = 0
        for t, tid in enumerate(r):
            if tid == image_token:
                k += 1
                index[b, t] = k
    e = sessions["embed"].run(None, {"input_ids": ids, "image_features": features, "image_index": index})[0]
    h = sessions["lower"].run(None, {"embeds": e})[0]
    logits = sessions["upper"].run(None, {"hidden": h, "last": np.array([len(r) - 1 for r in rows], np.int64)})[0]
    out = []
    for b, groups in enumerate(case["groups"]):
        scores = [max(float(logits[b, i]) for i in g) for g in groups]
        m = max(scores)
        ex = [math.exp(s - m) for s in scores]
        out.append([x / sum(ex) for x in ex])
    return out


for suffix in suffixes:
    sessions = {n: ort.InferenceSession(f"{onnx_dir}/{n}{suffix}.onnx", providers=["CPUExecutionProvider"])
                for n in ("embed", "lower", "upper", "vision")}
    worst, agree, total = 0.0, 0, 0
    for case in recorded:
        for r, g in zip(case["probs"], answer(sessions, case)):
            gap = max(abs(a - b) for a, b in zip(r, g))
            worst, agree, total = max(worst, gap), agree + int(np.argmax(r) == np.argmax(g)), total + 1
            print(f"  {suffix or 'fp32':5} {case['name']:6} gap={gap:.4f} ref={np.round(r, 3).tolist()} onnx={np.round(g, 3).tolist()}")
    print(f"{suffix or 'fp32'}: worst probability gap {worst:.4f}, top answer agrees on {agree}/{total}")
    del sessions
    gc.collect()
json.dump([{k: v for k, v in c.items() if k != "vision"} for c in recorded], open(f"{onnx_dir}/golden.json", "w"),
          ensure_ascii=False, indent=1)
