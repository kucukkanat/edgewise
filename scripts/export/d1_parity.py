"""Compare d1-omni ONNX graphs with the PyTorch original.

Usage: python d1_parity.py <hf-dir> <onnx-dir> [suffix ...]   (suffix "" is fp32, "_q8", "_q4", ...)

Builds inputs with d1's own prompt and media code, runs decide{suffix}.onnx (with vision and audio prefixes from
vision{suffix}.onnx and audio{suffix}.onnx), and prints the largest probability gap and top-answer agreement against
model.probabilities(). Also writes golden.json (token ids and reference probabilities) for the Edgewise test.
"""

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
mod = sys.modules[type(model).__module__]
prompt, vision, audio_mod = (sys.modules[mod.__name__.rsplit(".", 1)[0] + "." + n] for n in ("prompt", "vision", "audio"))
tok = model.tokenizer

refund = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
lane = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "tech": "bugs and crashes", "chat": "small talk"}}
urgency = {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "today", "blocking now"]}
many = {"type": "choice", "instructions": "Which language is this?",
        "criteria": {k: "" for k in ["English", "German", "French", "Spanish", "Italian", "Dutch", "Turkish", "Polish",
                                     "Swedish", "Czech", "Greek", "Finnish"]}}


def disc(size, color):
    img = Image.new("RGB", size, "white")
    w, h = size
    px = img.load()
    for y in range(h):
        for x in range(w):
            if (x - w / 2) ** 2 + (y - h / 2) ** 2 < (min(w, h) * 0.35) ** 2:
                px[x, y] = color
    return img


rng = np.random.default_rng(0)
tone = lambda s: (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(16000 * s)) / 16000) + 0.01 * rng.standard_normal(int(16000 * s))).astype(np.float32)  # noqa: E731
color = {"type": "choice", "instructions": "What color is the circle?", "criteria": {"red": "", "blue": "", "green": ""}}
circle = {"type": "noul", "instructions": "Is there a circle in the image?"}
topic = {"type": "choice", "instructions": "What is this sound?", "criteria": {"speech": "", "music": "", "a tone": ""}}

cases = [
    ("text", "I was charged twice for my order, please refund one of the payments.", [refund, lane, urgency], None, None),
    ("json", {"message": "The app crashes when I open settings.", "plan": "pro"}, [refund, lane], None, None),
    ("many", "Ich möchte mein Geld zurück, bitte. 😡 <|mask|>", [many, refund], None, None),
    ("image", None, [color, circle], [disc((256, 256), (255, 0, 0))], None),
    ("tiled", "A product photo.", [color], [disc((1600, 700), (0, 0, 255))], None),
    ("audio3s", "Voice note.", [topic], None, tone(3)),
    ("audio12s", None, [topic, circle], None, tone(12.3)),
]


def session(name, suffix):
    return ort.InferenceSession(f"{onnx_dir}/{name}{suffix}.onnx", providers=["CPUExecutionProvider"])


def resize_weights(out_size, in_size=16):
    eye = torch.eye(in_size)[None, None]
    return torch.nn.functional.interpolate(eye, size=(in_size, out_size), mode="bilinear", align_corners=False,
                                           antialias=True)[0, 0].T.numpy()


def image_prefix(s, images):
    out = []
    for image in images:
        inputs = vision.preprocess(image)
        for i, (h, w) in enumerate(inputs["spatial_shapes"].tolist()):
            patches = inputs["pixel_values"][i, : h * w].reshape(1, h, w, -1).numpy()
            out.append(s.run(None, {"patches": patches, "rows": resize_weights(h), "cols": resize_weights(w)})[0][0])
    return np.concatenate(out)


def audio_prefix(s, clip):
    mel, frames = model.audio.frontend(audio_mod.waveform(clip))
    n = int(frames[0])
    x = s.run(None, {"input_features": mel.transpose(1, 2).numpy(), "length": np.array([n], np.int64)})[0][0]
    for _ in range(3):
        n = (n - 1) // 2 + 1
    return x[:n]


def onnx_probs(sessions, state, questions, images, clip):
    cfg = model.config
    if images:
        prefix, max_len, noul, spoken = image_prefix(sessions["vision"], images), cfg.image_text_length, mod.YES_NO, False
    elif clip is not None:
        prefix, max_len, noul, spoken = audio_prefix(sessions["audio"], clip), cfg.audio_text_length, mod.YES_NO, True
        state = {} if state is None else state
    else:
        prefix, max_len, noul, spoken = None, cfg.max_length, None, False
    p = 0 if prefix is None else len(prefix)
    max_len = min(max_len, cfg.max_length - p)
    qs = [prompt.as_question(q) for q in questions]
    rows = [prompt.encode(tok, "" if state is None else state, q, max_len, noul, spoken) for q in qs]
    B, T, K = len(rows), max(len(r[0]) for r in rows), max(len(r[1]) for r in rows)
    feeds = {"input_ids": np.zeros((B, T), np.int64), "attention_mask": np.zeros((B, T), np.int64),
             "marker_pos": np.zeros((B, K), np.int64), "marker_mask": np.zeros((B, K), np.int64),
             "qtype": np.array([prompt.QTYPES[q.type] for q in qs], np.int64)}
    for b, (ids, markers) in enumerate(rows):
        feeds["input_ids"][b, : len(ids)] = ids
        feeds["attention_mask"][b, : len(ids)] = 1
        feeds["marker_pos"][b, : len(markers)] = markers
        feeds["marker_mask"][b, : len(markers)] = 1
    pin = max(p, 1)
    feeds["prefix"] = np.zeros((B, pin, 1024), np.float32) if prefix is None else np.repeat(prefix[None], B, 0)
    feeds["prefix_mask"] = np.full((B, pin), 0 if prefix is None else 1, np.int64)
    logits = sessions["decide"].run(None, feeds)[0]
    out = []
    for b, q in enumerate(qs):
        z = logits[b, : q.options].astype(np.float64)
        if prefix is None:
            z = z / cfg.temperatures.get(prompt.temperature_key(q), cfg.temperatures.get(q.type, 1.0))
        e = np.exp(z - z.max())
        pr = (e / e.sum()).tolist()
        out.append(pr[::-1] if q.type == "noul" else pr)
    return out, rows


golden = []
refs = {}
for name, state, questions, images, clip in cases:
    refs[name] = model.probabilities(state, questions, images=images, audio=clip)
for suffix in suffixes:
    sessions = {n: session(n, suffix) for n in ("decide", "vision", "audio")}
    worst, agree, total = 0.0, 0, 0
    for name, state, questions, images, clip in cases:
        got, rows = onnx_probs(sessions, state, questions, images, clip)
        for r, g in zip(refs[name], got):
            gap = max(abs(a - b) for a, b in zip(r, g))
            worst = max(worst, gap)
            agree += int(np.argmax(r) == np.argmax(g))
            total += 1
            print(f"  {suffix or 'fp32':5} {name:9} gap={gap:.4f} ref={np.round(r, 3).tolist()} onnx={np.round(g, 3).tolist()}")
        if suffix == suffixes[0] and images is None and clip is None:
            golden.append({"name": name, "state": state, "questions": questions, "ids": [r[0] for r in rows],
                           "markers": [r[1] for r in rows], "probs": refs[name]})
    print(f"{suffix or 'fp32'}: worst probability gap {worst:.4f}, top answer agrees on {agree}/{total}")
json.dump(golden, open(f"{onnx_dir}/golden.json", "w"), ensure_ascii=False, indent=1)
assert not math.isnan(worst)
