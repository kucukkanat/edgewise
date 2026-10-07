"""Export LiquidAI/d1-3B to ONNX in parts that each fit in one file under 2 GB once quantized.

Usage: python d1_3b_export.py <hf-dir> <out-dir> [embed,lower,upper,vision]

d1-3B is LFM2.5-VL-3B (a causal LFM2 decoder and a SigLIP2 tower) post-trained to answer at the last position:
each option is a token (yes/no, a letter, a digit) and the answer is a softmax over those tokens' logits.

embed.onnx   input_ids [B,T] int64, image_features [M,2048] float32, image_index [B,T] int64
             -> embeds [B,T,2048]; position t takes image_features[image_index[t]] when that index is > 0, else the
             token embedding (row 0 of image_features is an unused placeholder, so M >= 1 even without images)
lower.onnx   embeds [B,T,2048] -> hidden [B,T,2048]                 decoder layers 0-14
upper.onnx   hidden [B,T,2048], last [B] int64 -> logits [B,128000]  layers 15-29, final norm, head at `last`
vision.onnx  patches [1,h,w,768] f32, rows [h,16], cols [w,16] -> features [1,h*w/4,2048]   one crop per run

Rows are right-padded; the decoder is causal, so padding after a row's last token never changes it.
Each part is exported in fp32 with external data; d1_quant.py then makes single-file graphs: 8-bit MatMuls, and a
4-bit embedding table (ONNX Runtime quantizes Gather only at 4 bits).
"""

import os
import sys

import torch
import torch.nn.functional as F
from transformers import Lfm2VlForConditionalGeneration

src, out = sys.argv[1], sys.argv[2]
parts = sys.argv[3].split(",") if len(sys.argv) > 3 else ["embed", "lower", "upper", "vision"]
os.makedirs(out, exist_ok=True)
model = Lfm2VlForConditionalGeneration.from_pretrained(src, dtype=torch.float32).eval()
lm = model.model.language_model
cfg = model.config.text_config
H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
HD = cfg.hidden_size // H
THETA = cfg.rope_parameters["rope_theta"]
SPLIT = len(lm.layers) // 2


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope(t):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    freqs = torch.arange(t, dtype=torch.float32)[:, None] * inv[None]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]


def attention(a, x, cos, sin):
    b, t, _ = x.shape
    q = a.q_layernorm(a.q_proj(x).view(b, t, H, HD)).transpose(1, 2)
    k = a.k_layernorm(a.k_proj(x).view(b, t, KV, HD)).transpose(1, 2)
    v = a.v_proj(x).view(b, t, KV, HD).transpose(1, 2)
    q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    k, v = k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return a.out_proj(y.transpose(1, 2).reshape(b, t, H * HD))


def short_conv(c, x):
    t = x.shape[1]
    bb, cc, xx = c.in_proj(x).transpose(-1, -2).chunk(3, dim=-2)
    y = c.conv(bb * xx)[..., :t]
    return c.out_proj((cc * y).transpose(-1, -2))


def layers(h, start, stop):
    cos, sin = rope(h.shape[1])
    for layer in lm.layers[start:stop]:
        x = layer.operator_norm(h)
        h = h + (attention(layer.self_attn, x, cos, sin) if layer.is_attention_layer else short_conv(layer.conv, x))
        h = h + layer.feed_forward(layer.ffn_norm(h))
    return h


class Embed(torch.nn.Module):
    def forward(self, input_ids, image_features, image_index):
        tokens = lm.embed_tokens(input_ids)
        return torch.where((image_index > 0)[..., None], image_features[image_index], tokens)


class Lower(torch.nn.Module):
    def forward(self, embeds):
        return layers(embeds, 0, SPLIT)


class Upper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The head is tied to the embedding; a pre-transposed copy makes it a MatMul (not a Gemm), which
        # MatMulNBits quantizes.
        self.head = torch.nn.Parameter(model.lm_head.weight.detach().T.contiguous(), requires_grad=False)

    def forward(self, hidden, last):
        h = layers(hidden, SPLIT, len(lm.layers))
        h = torch.gather(h, 1, last[:, None, None].expand(-1, 1, h.shape[-1]))
        return torch.matmul(lm.embedding_norm(h), self.head)[:, 0]


def resize_weights(out_size, in_size=16):
    eye = torch.eye(in_size)[None, None]
    return F.interpolate(eye, size=(in_size, out_size), mode="bilinear", align_corners=False, antialias=True)[0, 0].T


class VisionCrop(torch.nn.Module):
    def forward(self, patches, rows, cols):
        vt = model.model.vision_tower
        emb = vt.embeddings if hasattr(vt, "embeddings") else vt.vision_model.embeddings
        tower = vt if hasattr(vt, "embeddings") else vt.vision_model
        _, h, w, c = patches.shape
        x = emb.patch_embedding(patches.reshape(1, h * w, c))
        n = emb.position_embedding_size
        grid = emb.position_embedding.weight.reshape(n, n, -1)
        x = x + torch.einsum("hi,ijc,wj->hwc", rows, grid, cols).reshape(1, h * w, -1)
        x = tower.post_layernorm(tower.encoder(x, None).last_hidden_state)
        return model.model.multi_modal_projector(x.reshape(1, h, w, -1)).reshape(1, -1, cfg.hidden_size)


def save(module, args, name, inputs, outputs, dims):
    path = f"{out}/{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(module, args, path, dynamo=True, opset_version=18, input_names=inputs, output_names=outputs,
                          dynamic_shapes=dims, external_data=True)
    print("saved", path)


with torch.no_grad():
    # The plain forward matches HF's on a sample, checked before any export.
    ids = torch.tensor([[1, 1404, 22, 4410, 892, 7, 2], [1, 315, 9001, 23, 5, 0, 0]])
    ref = model(input_ids=ids[:1]).logits[0, -1]
    mine = Upper()(Lower()(Embed()(ids, torch.zeros(1, cfg.hidden_size), torch.zeros_like(ids))), torch.tensor([6, 4]))
    print("plain forward vs HF, max logit gap", (ref - mine[0]).abs().max().item())

B, T = torch.export.Dim("batch", min=1, max=512), torch.export.Dim("seq", min=2, max=32768)
sample_ids = torch.randint(10, 1000, (2, 12))
if "embed" in parts:
    M = torch.export.Dim("images", min=1, max=40000)
    save(Embed(), (sample_ids, torch.randn(5, cfg.hidden_size), torch.zeros(2, 12, dtype=torch.long)), "embed",
         ["input_ids", "image_features", "image_index"], ["embeds"],
         {"input_ids": {0: B, 1: T}, "image_features": {0: M}, "image_index": {0: B, 1: T}})
if "lower" in parts:
    save(Lower(), (torch.randn(2, 12, cfg.hidden_size),), "lower", ["embeds"], ["hidden"], {"embeds": {0: B, 1: T}})
if "upper" in parts:
    save(Upper(), (torch.randn(2, 12, cfg.hidden_size), torch.tensor([11, 5])), "upper", ["hidden", "last"], ["logits"],
         {"hidden": {0: B, 1: T}, "last": {0: B}})
if "vision" in parts:
    hd = torch.export.Dim("h", min=2, max=64)
    wd = torch.export.Dim("w", min=2, max=64)
    # A patch is 3 × 16 × 16 = 768 RGB values, whatever the tower's width.
    save(VisionCrop(), (torch.randn(1, 24, 20, 768), resize_weights(24), resize_weights(20)), "vision",
         ["patches", "rows", "cols"], ["features"], {"patches": {1: hd, 2: wd}, "rows": {0: hd}, "cols": {0: wd}})
