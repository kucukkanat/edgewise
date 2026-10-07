"""Export LiquidAI/d1-omni-600M to ONNX: the decision graph (trunk + head) and the audio encoder.

Usage: python d1_export.py <hf-dir> <out-dir> [decide,vision,audio]

decide.onnx
  input_ids [B,T] int64, attention_mask [B,T] int64      one row per question, text right-padded
  prefix [B,P,1024] float32, prefix_mask [B,P] int64      media embeddings; text-only calls pass P=1, mask 0
  marker_pos [B,K] int64, marker_mask [B,K] int64        the <|mask|> position of each option, in text coordinates
  qtype [B] int64                                        0 choice, 1 score, 2 noul
  -> logits [B,K]                                        softmax over the first `options` entries (after temperature)

A fully masked one-position prefix gives the text-only answer: its key is masked out of attention, the short
convolution zeroes it, and RoPE is relative, so shifting the text by one position changes nothing.

vision.onnx
  patches [1,h,w,768] float32  one crop's 16 px patches (RGB in [-1, 1]) laid out as its patch grid
  rows [h,16], cols [w,16]     antialiased bilinear resize weights (PyTorch's, see resizeWeights in src/backends/d1.ts)
  -> features [1,h*w/4,1024]   that crop's prefix embeddings; concatenate crops in order (tiles, then thumbnail)

SigLIP2 resizes its 16x16 position-embedding grid to each crop with an antialiased bilinear resize. That resize is
linear and separable, so it is rows @ grid @ cols^T; passing the weights keeps the graph to einsum, because ONNX
Runtime's WebGPU provider does not implement Resize with antialias.

audio.onnx
  input_features [1,F,128] float32 (Parakeet/NeMo log-mel), length [1] int64 (valid frames)
  -> prefix [1,F8,1024]; keep the first L8 rows, L8 = three rounds of floor((L - 1) / 2) + 1

d1 fine-tuned its vision tower (every tensor differs from LFM2.5-VL-450M's), so the vision graph is exported here too.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel

src, out = sys.argv[1], sys.argv[2]
parts = sys.argv[3].split(",") if len(sys.argv) > 3 else ["decide", "vision", "audio"]
os.makedirs(out, exist_ok=True)
model = AutoModel.from_pretrained(src, trust_remote_code=True, dtype=torch.float32).eval()


def encoder_layer(layer, x, key_pad):
    """nn.TransformerEncoderLayer (norm_first, ReLU) written out, so export never takes the fused fast path."""
    attn = layer.self_attn
    b, t, d = x.shape
    h = attn.num_heads
    y = layer.norm1(x)
    q, k, v = F.linear(y, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q, k, v = (z.view(b, t, h, d // h).transpose(1, 2) for z in (q, k, v))
    mask = torch.zeros(b, 1, 1, t, dtype=x.dtype).masked_fill(key_pad[:, None, None, :], float("-inf"))
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = x + attn.out_proj(y.transpose(1, 2).reshape(b, t, d))
    return x + layer.linear2(F.relu(layer.linear1(layer.norm2(x))))


class Decide(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.trunk, self.head = m.encoder, m.head

    def forward(self, input_ids, attention_mask, prefix, prefix_mask, marker_pos, marker_mask, qtype):
        p = prefix.shape[1]
        h = torch.cat([prefix, self.trunk.embed_tokens(input_ids)], dim=1)
        pad = torch.cat([prefix_mask, attention_mask], dim=1).bool()
        offsets = torch.ones_like(prefix_mask).sum(1)  # P per row, from tensors so the trace keeps it dynamic
        text = self.trunk(h, pad, offsets)[:, p:]
        head = self.head
        x = text + head.type_emb(qtype)[:, None, :]
        for layer in head.head.layers:
            x = encoder_layer(layer, x, ~attention_mask.bool())
        g = torch.gather(x, 1, marker_pos[:, :, None].expand(-1, -1, x.shape[-1]))
        return head.scorer(g).squeeze(-1).masked_fill(~marker_mask.bool(), -1e4)


def pos_emb(self, t, device):
    """Conformer.pos_emb with the same values, built so its length is exactly 2t - 1 to the shape solver;
    arange(t - 1, -t, -1) is not, which makes torch.export pin the clip length."""
    positions = (t - 1 - torch.arange(2 * t - 1, dtype=torch.float32))[:, None]
    div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
    angles = positions * div
    return torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(2 * t - 1, self.d_model)[None].to(device)


def rel_attention(self, x, pos_emb, mask):
    """RelPositionAttention.forward with the Transformer-XL shift as a gather (out[i, j] = bd[i, t - 1 - i + j]);
    the original pad-and-view trick hides the clip length from torch.export."""
    b, t, _ = x.shape
    q = self.linear_q(x).view(b, t, self.h, self.d_k)
    k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
    v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
    p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)
    ac = torch.matmul((q + self.pos_bias_u).transpose(1, 2), k.transpose(-2, -1))
    bd = torch.matmul((q + self.pos_bias_v).transpose(1, 2), p.transpose(-2, -1))
    i = torch.arange(t)
    idx = (t - 1 - i[:, None] + i[None, :]).expand(bd.shape[0], bd.shape[1], t, t)
    scores = (ac + torch.gather(bd, -1, idx)) / math.sqrt(self.d_k)
    scores = scores.masked_fill(mask[:, None], -10000.0)
    attn = torch.softmax(scores, dim=-1).masked_fill(mask[:, None], 0.0)
    return self.linear_out(torch.matmul(attn, v).transpose(1, 2).reshape(b, t, self.h * self.d_k))


class AudioEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.audio = m.audio
        encoder = m.audio.encoder
        mel = torch.randn(1, 777, 128)
        with torch.no_grad():
            before = encoder(mel.transpose(1, 2), torch.tensor([770]))[0]
            type(encoder).pos_emb = pos_emb
            type(encoder.layers[0].self_attn).forward = rel_attention
            after = encoder(mel.transpose(1, 2), torch.tensor([770]))[0]
        print("export-friendly conformer, max gap", (before - after).abs().max().item())

    def forward(self, input_features, length):
        a = self.audio
        x, _ = a.encoder(input_features.transpose(1, 2), length)
        return a.residual(a.adapter(x))


def resize_weights(out_size, in_size=16):
    """The (out, in) matrix of F.interpolate(mode="bilinear", antialias=True, align_corners=False) along one axis."""
    eye = torch.eye(in_size)[None, None]
    return F.interpolate(eye, size=(in_size, out_size), mode="bilinear", align_corners=False, antialias=True)[0, 0].T


class VisionCrop(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.vm, self.projector = m.vision.tower, m.vision.projector

    def forward(self, patches, rows, cols):
        vm = self.vm
        _, h, w, c = patches.shape
        x = vm.embeddings.patch_embedding(patches.reshape(1, h * w, c))
        n = vm.embeddings.position_embedding_size
        grid = vm.embeddings.position_embedding.weight.reshape(n, n, -1)
        x = x + torch.einsum("hi,ijc,wj->hwc", rows, grid, cols).reshape(1, h * w, -1)
        x = vm.post_layernorm(vm.encoder(x, None).last_hidden_state)
        return self.projector(x.reshape(1, h, w, -1))


def export(module, args, path, names, outputs, dynamic):
    with torch.no_grad():
        torch.onnx.export(module, args, path, dynamo=False, opset_version=18, input_names=names, output_names=outputs,
                          dynamic_axes=dynamic, external_data=False)
    print("saved", path, os.path.getsize(path) / 1e6, "MB")


if "decide" in parts:
    B, T, P, K = 2, 40, 3, 4
    args = (torch.randint(10, 1000, (B, T)), torch.ones(B, T, dtype=torch.long), torch.randn(B, P, 1024),
            torch.ones(B, P, dtype=torch.long), torch.tensor([[5, 9, 13, 17]] * B), torch.ones(B, K, dtype=torch.long),
            torch.tensor([0, 2]))
    export(Decide(model).eval(), args, f"{out}/decide.onnx",
           ["input_ids", "attention_mask", "prefix", "prefix_mask", "marker_pos", "marker_mask", "qtype"], ["logits"],
           {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
            "prefix": {0: "batch", 1: "media"}, "prefix_mask": {0: "batch", 1: "media"},
            "marker_pos": {0: "batch", 1: "options"}, "marker_mask": {0: "batch", 1: "options"},
            "qtype": {0: "batch"}, "logits": {0: "batch", 1: "options"}})

if "vision" in parts:
    crop = VisionCrop(model).eval()
    with torch.no_grad():
        # The einsum matches SigLIP2's own resize.
        p = torch.randn(1, 24, 20, 768)
        emb = model.vision.tower.embeddings
        grid = emb.position_embedding.weight.reshape(16, 16, -1)
        ref = emb.resize_positional_embeddings(grid, torch.tensor([[24, 20]]), 480)
        mine = torch.einsum("hi,ijc,wj->hwc", resize_weights(24), grid, resize_weights(20)).reshape(1, 480, -1)
        print("position resize as einsum, max gap", (ref - mine).abs().max().item())
        dims = {"patches": {1: torch.export.Dim("h", min=2, max=64), 2: torch.export.Dim("w", min=2, max=64)},
                "rows": {0: torch.export.Dim("h", min=2, max=64)}, "cols": {0: torch.export.Dim("w", min=2, max=64)}}
        torch.onnx.export(crop, (p, resize_weights(24), resize_weights(20)), f"{out}/vision.onnx", dynamo=True,
                          opset_version=18, input_names=["patches", "rows", "cols"], output_names=["features"],
                          dynamic_shapes=dims, external_data=False)
    print("saved", f"{out}/vision.onnx", os.path.getsize(f"{out}/vision.onnx") / 1e6, "MB")

if "audio" in parts:
    # dynamo, because the TorchScript tracer bakes the conformer's relative-position table to the traced length.
    frames = torch.export.Dim("frames", min=50, max=3001)
    with torch.no_grad():
        torch.onnx.export(AudioEncoder(model).eval(), (torch.randn(1, 301, 128), torch.tensor([300])), f"{out}/audio.onnx",
                          dynamo=True, opset_version=18, input_names=["input_features", "length"],
                          output_names=["prefix"], dynamic_shapes={"input_features": {1: frames}, "length": None},
                          external_data=False)
    print("saved", f"{out}/audio.onnx", os.path.getsize(f"{out}/audio.onnx") / 1e6, "MB")
