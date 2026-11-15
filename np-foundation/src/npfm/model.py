"""Original compact sequence/graph research model; no third-party pretrained weights."""
from __future__ import annotations

import torch
from torch import nn
from .features import ATOM_FEATURE_DIM


class GraphEncoder(nn.Module):
    def __init__(self, d_model, num_layers=2):
        super().__init__()
        self.input = nn.Linear(ATOM_FEATURE_DIM, d_model)
        self.layers = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))

    def forward(self, graph):
        x, adjacency, mask = graph["atoms"], graph["adjacency"], graph["mask"]
        x = torch.nn.functional.gelu(self.input(x))
        adjacency = adjacency + torch.diag_embed(mask.to(adjacency.dtype))
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1.)
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + torch.nn.functional.gelu(layer(torch.bmm(adjacency, x))))
            x = x * mask.unsqueeze(-1)
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


class NaturalProductModel(nn.Module):
    def __init__(self, vocab_size, max_seq_length=512, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=.1, graph_layers=2, use_graph=False,
                 tasks=None, pad_id=0):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead.")
        self.max_seq_length, self.pad_id, self.use_graph = max_seq_length, pad_id, use_graph
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                          batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.sequence_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        if use_graph:
            self.graph_encoder = GraphEncoder(d_model, graph_layers)
            self.fusion = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.heads = nn.ModuleDict({t["name"]: nn.Linear(d_model, int(t.get("output_dim", 1)))
                                    for t in (tasks or []) if t.get("trainable", True)})

    def forward(self, ids, graph=None, causal=True, lm_only=False):
        if ids.ndim != 2 or ids.shape[1] == 0 or ids.shape[1] > self.max_seq_length:
            raise ValueError("Token batch must be nonempty and within max_seq_length.")
        valid = ids.ne(self.pad_id)
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        attention_mask = torch.ones((ids.shape[1], ids.shape[1]), device=ids.device, dtype=torch.bool).triu(1) if causal else None
        x = self.sequence_norm(self.encoder(x, mask=attention_mask, src_key_padding_mask=~valid))
        sequence = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        result = {"lm_logits": self.lm_head(x), "sequence_embedding": sequence}
        if lm_only:
            return result
        if self.use_graph:
            if graph is None:
                raise ValueError("This checkpoint requires real RDKit graphs.")
            graph_embedding = self.graph_encoder(graph)
            embedding = self.fusion(torch.cat([sequence, graph_embedding], dim=-1))
        else:
            graph_embedding, embedding = None, sequence
        result.update(graph_embedding=graph_embedding, embedding=embedding,
                      tasks={name: head(embedding) for name, head in self.heads.items()})
        return result
