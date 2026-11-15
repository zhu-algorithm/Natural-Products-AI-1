"""NP-GPT: an original PyTorch implementation of a causal molecular Transformer.

RoPE, RMSNorm, SwiGLU, grouped-query attention and Transformer attention are
established methods, not new mathematical inventions of this project. This file
implements them directly; it contains no third-party model code or weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NPTransformerConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: Optional[int] = 2
    ffn_hidden_dim: Optional[int] = None
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    pad_token_id: Optional[int] = 0
    tie_embeddings: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least two.")
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if isinstance(self.n_kv_heads, bool) or not isinstance(self.n_kv_heads, int) or self.n_kv_heads < 1:
            raise ValueError("n_kv_heads must be a positive integer or None.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension.")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = 64 * math.ceil((8 * self.d_model / 3) / 64)
        if isinstance(self.ffn_hidden_dim, bool) or not isinstance(self.ffn_hidden_dim, int) or self.ffn_hidden_dim < 1:
            raise ValueError("ffn_hidden_dim must be a positive integer.")
        for name in ("rope_theta", "norm_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1).")
        if self.pad_token_id is not None and (isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int) or not 0 <= self.pad_token_id < self.vocab_size):
            raise ValueError("pad_token_id must be a vocabulary index or None.")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be bool.")

    def to_dict(self):
        """Return only JSON-serializable constructor fields, never derived state."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Model configuration must be a dictionary.")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("Unknown model configuration fields: " + ", ".join(sorted(unknown)))
        return cls(**values)

    @classmethod
    def from_preset(cls, preset, vocab_size, **overrides):
        """Architecture sizes, not pretrained model claims; vocabulary affects counts."""
        presets = {
            "smoke": {"d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
                      "ffn_hidden_dim": 192, "max_seq_len": 512},
            "small": {"d_model": 192, "n_layers": 4, "n_heads": 6, "n_kv_heads": 2,
                      "ffn_hidden_dim": 512, "max_seq_len": 1024},
            "base": {"d_model": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 4,
                     "ffn_hidden_dim": 1408, "max_seq_len": 2048},
        }
        if preset not in presets:
            raise ValueError("preset must be smoke, small or base.")
        return cls(vocab_size=vocab_size, **{**presets[preset], **overrides})


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, hidden):
        # Accumulate normalization in float32, including under mixed precision.
        values = hidden.float()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(hidden.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        inverse_frequency = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        # Derived from config; no large or mutable position table in checkpoints.
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, positions):
        angles = positions.float().unsqueeze(-1) * self.inverse_frequency.float()
        return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)


def _rotate_pairs(values, cosine, sine):
    cosine, sine = cosine.to(values.dtype), sine.to(values.dtype)
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, rotary, allowed_attention, causal, past, use_cache):
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        query, key = _rotate_pairs(query, *rotary), _rotate_pairs(key, *rotary)
        if past is not None:
            if past[0].dtype != key.dtype or past[1].dtype != value.dtype:
                raise ValueError("KV cache dtype must match current attention projection dtype.")
            key, value = torch.cat((past[0], key), dim=2), torch.cat((past[1], value), dim=2)
        present = (key, value) if use_cache else None
        # Keep the cache compact. Repeat only for attention computation, avoiding a
        # dependency on backend-specific native GQA support in PyTorch SDPA.
        groups = self.n_heads // self.n_kv_heads
        if groups > 1:
            attention_key = key.repeat_interleave(groups, dim=1)
            attention_value = value.repeat_interleave(groups, dim=1)
        else:
            attention_key, attention_value = key, value
        attended = F.scaled_dot_product_attention(
            query, attention_key, attention_value, attn_mask=allowed_attention,
            dropout_p=self.dropout if self.training else 0.0, is_causal=causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.output_dropout(self.o_proj(attended)), present


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, hidden, rotary, allowed_attention, causal, past, use_cache):
        update, present = self.attention(self.attention_norm(hidden), rotary, allowed_attention, causal, past, use_cache)
        hidden = hidden + update
        return hidden + self.ffn(self.ffn_norm(hidden)), present


class NPTransformer(nn.Module):
    """Causal language model with explicitly aligned next-token supervision.

    ``targets`` has the same shape as ``input_ids`` and is ALREADY shifted, e.g.
    input_ids=ids[:, :-1], targets=ids[:, 1:]. Targets equal to pad_token_id or
    -100 are ignored, as are masked query positions. With no valid targets the
    loss is differentiable zero, never NaN.

    A cache is a tuple of (rotated_key, value) per layer, each with shape
    [batch, n_kv_heads, prefix_length, head_dim]. Subsequent calls pass only NEW
    input tokens. When using padding, attention_mask must cover the entire
    cached prefix plus new tokens and retain the same prefix mask. Without an
    attention_mask all positions are treated as valid. Caches are not mutated
    or silently detached; generation callers should use torch.no_grad().
    """
    def __init__(self, config):
        super().__init__()
        if not isinstance(config, NPTransformerConfig):
            raise TypeError("config must be NPTransformerConfig.")
        # Own a validated copy so external config mutation cannot change shapes.
        self.config = NPTransformerConfig.from_dict(config.to_dict())
        config = self.config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.rotary_embedding = RotaryEmbedding(config.d_model // config.n_heads, config.rope_theta)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=.02)
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.config.pad_token_id].zero_()
        if not self.config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, std=.02)
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
                elif isinstance(module, RMSNorm):
                    nn.init.ones_(module.weight)
            # Scale residual branches with depth to moderate initial activations.
            scale = .02 / math.sqrt(2 * self.config.n_layers)
            nn.init.normal_(layer.attention.o_proj.weight, std=scale)
            nn.init.normal_(layer.ffn.down_proj.weight, std=scale)
        nn.init.ones_(self.final_norm.weight)

    def num_parameters(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def _validate_cache(self, past_key_values, batch, device):
        if past_key_values is None:
            return 0
        if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) != self.config.n_layers:
            raise ValueError("past_key_values must have one (key, value) pair per decoder layer.")
        prefix_length = None
        for pair in past_key_values:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each cache layer must contain (key, value).")
            key, value = pair
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor) or key.ndim != 4 or value.shape != key.shape:
                raise ValueError("Cache key/value tensors must have matching four-dimensional shapes.")
            if key.shape[0] != batch or key.shape[1] != self.config.n_kv_heads or key.shape[3] != self.config.d_model // self.config.n_heads:
                raise ValueError("KV cache batch/head dimensions do not match this model/input.")
            if key.device != device or value.device != device or not key.is_floating_point() or not value.is_floating_point():
                raise ValueError("KV cache must contain floating tensors on the input device.")
            if prefix_length is None:
                prefix_length = key.shape[2]
            elif prefix_length != key.shape[2]:
                raise ValueError("All KV cache layers must have the same prefix length.")
        return prefix_length

    def _run_transformer(self, input_ids, attention_mask, past_key_values, use_cache):
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("input_ids must be a [batch, length] integer tensor.")
        batch, length = input_ids.shape
        if batch < 1 or length < 1:
            raise ValueError("input_ids must have a nonempty batch and sequence.")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("input_ids must be on the same device as the model.")
        if bool((input_ids < 0).any()) or bool((input_ids >= self.config.vocab_size).any()):
            raise ValueError("input_ids contain a token outside the configured vocabulary.")
        prefix_length = self._validate_cache(past_key_values, batch, input_ids.device)
        total_length = prefix_length + length
        if total_length > self.config.max_seq_len:
            raise ValueError("Prefix plus new tokens exceed max_seq_len; context is never silently truncated.")
        valid = None
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (batch, total_length):
                raise ValueError("attention_mask must cover [batch, cached_prefix_length + new_length].")
            if attention_mask.device != input_ids.device:
                raise ValueError("attention_mask must be on the input device.")
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must be binary, with 1/True meaning a valid token.")
            valid = attention_mask.bool()
            # Padded positions do not advance RoPE positions. This supports both
            # left/right padded prefixes when the full mask is supplied with cache.
            positions = (valid.long().cumsum(-1) - 1).clamp_min(0)[:, prefix_length:]
        else:
            positions = torch.arange(prefix_length, total_length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        if valid is None and prefix_length == 0:
            allowed_attention, causal = None, True
        elif valid is None and length == 1:
            # The new single query may attend the entire existing prefix.
            allowed_attention, causal = None, False
        else:
            query_positions = torch.arange(prefix_length, total_length, device=input_ids.device)
            key_positions = torch.arange(total_length, device=input_ids.device)
            allowed_attention = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
            if valid is not None:
                allowed_attention = allowed_attention & valid[:, None, None, :]
            # SDPA's rectangular is_causal mask is upper-left aligned. Cached
            # multi-token chunks require the explicit prefix-offset mask above.
            causal = False
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        rotary = self.rotary_embedding(positions)
        new_cache = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            past = past_key_values[index] if past_key_values is not None else None
            hidden, present = layer(hidden, rotary, allowed_attention, causal, past, use_cache)
            if use_cache:
                new_cache.append(present)
        return self.final_norm(hidden), tuple(new_cache) if use_cache else None, valid

    def forward(self, input_ids, targets=None, attention_mask=None, past_key_values=None,
                use_cache=False, output_hidden_states=False):
        hidden, present, valid = self._run_transformer(input_ids, attention_mask, past_key_values, use_cache)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            if not isinstance(targets, torch.Tensor) or targets.shape != input_ids.shape or targets.dtype not in {torch.int32, torch.int64}:
                raise ValueError("targets must be an already shifted integer tensor with the same shape as input_ids.")
            if targets.device != input_ids.device:
                raise ValueError("targets must be on the input device.")
            if bool(((targets < 0) & (targets != -100)).any()) or bool((targets >= self.config.vocab_size).any()):
                raise ValueError("Targets must be vocabulary indices or the ignore index -100.")
            labels = targets.long().clone()
            if self.config.pad_token_id is not None:
                labels.masked_fill_(labels == self.config.pad_token_id, -100)
            if valid is not None:
                labels.masked_fill_(~valid[:, -input_ids.shape[1]:], -100)
            if bool(labels.ne(-100).any()):
                loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size), labels.reshape(-1), ignore_index=-100)
            else:
                loss = logits.float().sum() * 0.0
        result = {"logits": logits, "loss": loss, "past_key_values": present}
        if output_hidden_states:
            # Only the final normalized states; no per-layer activation history.
            result["hidden_states"] = hidden
        return result

    def encode(self, input_ids, attention_mask=None):
        """Final causal token representations for separately supervised tasks.

        Returns [batch, length, d_model] without computing the vocabulary logits.
        Callers must mask padding when pooling; no disease/property head is added.
        """
        return self._run_transformer(input_ids, attention_mask, None, False)[0]
