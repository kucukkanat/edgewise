"""LFM2 backbone with bidirectional attention + non-causal short-conv.

Wired into the HF repo via `auto_map` in config.json so that

AutoModel.from_pretrained(repo, trust_remote_code=True)
AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)

both return a model with the encoder-style patches already applied.

Supports `attn_implementation` in {"eager", "sdpa", "flash_attention_2"}:

eager/sdpa consume a 4D additive pad-only mask and reproduce the exact
training-time behavior; flash_attention_2 receives the 2D padding mask (or
None) and runs the kernel non-causally via `Lfm2Attention.is_causal = False`,
yielding outputs equivalent to the unpadded forward.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.lfm2 import modeling_lfm2 as _lfm2_mod
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2Model,
    Lfm2PreTrainedModel,
    Lfm2ShortConv,
    apply_mask_to_padding_states,
)


def _bidirectional_mask(
    config,
    input_embeds: torch.Tensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    past_key_values=None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    # transformers has renamed the embeds kwarg across versions
    # (input_embeds <-> inputs_embeds); accept either to stay forward-compatible.
    if input_embeds is None:
        input_embeds = kwargs.get("inputs_embeds")

    if config._attn_implementation == "flash_attention_2":
        # FA2 only uses the 2D padding mask to unpad sequences; causality is
        # controlled by `Lfm2Attention.is_causal` (set to False below).
        if attention_mask is not None and not attention_mask.all():
            return attention_mask
        return None

    device = input_embeds.device
    dtype = input_embeds.dtype
    bsz, q_len = input_embeds.shape[:2]
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    kv_len = past + q_len

    mask = torch.zeros((bsz, 1, q_len, kv_len), device=device, dtype=dtype)
    if attention_mask is not None:
        cur_len = attention_mask.size(-1)
        key_pad_flags = (attention_mask == 0).to(device=device, dtype=torch.float32)
        pad_vec = torch.zeros((bsz, kv_len), device=device, dtype=torch.float32)
        if cur_len > 0:
            pad_vec[:, past:past + cur_len] = key_pad_flags * -1e9
        mask = mask + pad_vec.to(dtype)[:, None, None, :]
    return mask


def _noncausal_shortconv_forward(
    self,
    hidden_states: torch.Tensor,
    past_key_values=None,
    cache_position=None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    x = apply_mask_to_padding_states(hidden_states, attention_mask)

    BCx = self.in_proj(x).transpose(-1, -2)
    B, C, x = BCx.chunk(3, dim=-2)
    Bx = B * x

    k = self.conv.weight.shape[-1]
    pad = k // 2
    conv_out = F.conv1d(
        Bx, weight=self.conv.weight, bias=self.conv.bias,
        stride=1, padding=pad, dilation=1, groups=Bx.shape[1],
    )
    if conv_out.shape[-1] > Bx.shape[-1]:
        conv_out = conv_out[..., :Bx.shape[-1]]
    elif conv_out.shape[-1] < Bx.shape[-1]:
        conv_out = F.pad(conv_out, (0, Bx.shape[-1] - conv_out.shape[-1]))

    y = C * conv_out
    y = y.transpose(-1, -2).contiguous()
    return self.out_proj(y)


def _shortconv_forward(self, *args, **kwargs):
    return self.slow_forward(*args, **kwargs)


_PATCHED = False


def _install_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _lfm2_mod.create_causal_mask = _bidirectional_mask
    Lfm2ShortConv.slow_forward = _noncausal_shortconv_forward
    Lfm2ShortConv.forward = _shortconv_forward
    _PATCHED = True


_install_patches()


def _set_attention_noncausal(model) -> None:
    for module in model.modules():
        if isinstance(module, Lfm2Attention):
            module.is_causal = False


class Lfm2BidirectionalModel(Lfm2Model):
    """LFM2 patched for encoder-style use: 
    full bidirectional attention + non-causal short-conv."""

    def __init__(self, config):
        _install_patches()
        super().__init__(config)
        _set_attention_noncausal(self)


class Lfm2BidirectionalForMaskedLM(Lfm2PreTrainedModel):
    """LFM2 bidirectional encoder with a tied masked-LM head."""

    config_class = Lfm2Config
    base_model_prefix = "lfm2"
    _tied_weights_keys = {"lm_head.weight": "lfm2.embed_tokens.weight"}

    def __init__(self, config: Lfm2Config):
        _install_patches()
        config = type(config).from_dict({**config.to_dict(), "use_cache": False})
        super().__init__(config)
        self.lfm2 = Lfm2BidirectionalModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.lm_head.weight = self.lfm2.embed_tokens.weight

    def get_input_embeddings(self):
        return self.lfm2.embed_tokens

    def set_input_embeddings(self, value):
        self.lfm2.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        return_dict = True if return_dict is None else return_dict
        outputs = self.lfm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            out = (logits,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Lfm2BidirForSequenceRouting(Lfm2PreTrainedModel):
    """Zero-shot prompt router built on the bidirectional LFM2 encoder."""

    config_class = Lfm2Config
    base_model_prefix = "lfm2"

    def __init__(self, config: Lfm2Config):
        _install_patches()
        config = type(config).from_dict({**config.to_dict(), "use_cache": False})
        super().__init__(config)
        self.lfm2 = Lfm2BidirectionalModel(config)
        proj_dim = getattr(config, "rule_proj_dim", 256)
        self.tok_proj = nn.Linear(config.hidden_size, proj_dim)
        self.rule_proj = nn.Linear(config.hidden_size, proj_dim)
        self.score_bias = nn.Parameter(torch.tensor(0.0))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.post_init()

    def get_input_embeddings(self):
        return self.lfm2.embed_tokens

    def set_input_embeddings(self, value):
        self.lfm2.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_pool: Optional[torch.Tensor] = None,
        category_pool: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        outputs = self.lfm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        text_rep = torch.bmm(text_pool, hidden).squeeze(1)
        category_rep = torch.bmm(category_pool, hidden)
        query = F.normalize(self.tok_proj(text_rep), dim=-1)
        categories = F.normalize(self.rule_proj(category_rep), dim=-1)
        scale = torch.clamp(self.logit_scale.exp(), max=30.0)
        logits = torch.einsum("bd,brd->br", query, categories) * scale + self.score_bias
        return {"logits": logits}

    @staticmethod
    def _prefix(routes):
        body = "\n".join(f"- {route}" for route in routes) if routes else "- (none)"
        return f"Categories:\n{body}\n\nText:\n"

    @staticmethod
    def _category_ranges(routes):
        ranges = []
        pos = len("Categories:\n")
        for route in routes:
            start = pos + 2
            end = start + len(route)
            ranges.append((start, end))
            pos = end + 1
        return ranges

    @torch.no_grad()
    def route(self, text, routes, tokenizer, threshold=None):
        prefix = self._prefix(routes)
        full_text = prefix + text
        enc = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}

        text_start = len(prefix)
        text_idxs = [
            i for i, (start, end) in enumerate(offsets)
            if end > text_start and start != end
        ]
        text_pool = torch.zeros(1, 1, len(offsets), device=self.device)
        if text_idxs:
            text_pool[0, 0, text_idxs] = 1 / len(text_idxs)

        category_pool = torch.zeros(1, len(routes), len(offsets), device=self.device)
        for route_idx, (start, end) in enumerate(self._category_ranges(routes)):
            token_idxs = [
                i for i, (tok_start, tok_end) in enumerate(offsets)
                if tok_start < end and tok_end > start and tok_start != tok_end
            ]
            if token_idxs:
                category_pool[0, route_idx, token_idxs] = 1 / len(token_idxs)

        logits = self(**enc, text_pool=text_pool, category_pool=category_pool)["logits"][0]
        probs = logits.softmax(dim=-1).detach().cpu()
        results = [
            {"route": route, "score": float(prob)}
            for route, prob in zip(routes, probs)
            if threshold is None or prob >= threshold
        ]
        return sorted(results, key=lambda item: item["score"], reverse=True)
