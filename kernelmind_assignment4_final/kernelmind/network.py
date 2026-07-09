from __future__ import annotations

import math

import torch
from torch import nn


class DirectSchedulerNet(nn.Module):
    """Self-attention Q-network that produces one scalar Q-value per queue slot.

    This uses an explicit multi-head scaled dot-product attention block instead
    of nn.MultiheadAttention, which keeps the masking logic visible and easy to
    audit for this assignment.
    """

    def __init__(
        self,
        max_queue_size: int,
        feature_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.max_queue_size = max_queue_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_projection = nn.Linear(feature_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_queue_size, embed_dim))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.q_head = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, E] -> [B, H, N, D]
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _attention(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = ~valid_mask[:, None, None, :]  # True where a key is padding
        scores = scores.masked_fill(key_mask, -1e9)
        scores = torch.clamp(scores, min=-1e9, max=50.0)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # [B, H, N, D]
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return self.out_proj(out)

    def forward(self, state: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Return masked Q-values.

        state shape: [batch, N, feature_dim]
        valid_mask shape: [batch, N], True for real processes and False for padding.
        """
        original_valid_mask = valid_mask.bool()
        attn_valid_mask = original_valid_mask.clone()
        empty_rows = ~attn_valid_mask.any(dim=1)
        if empty_rows.any():
            attn_valid_mask[empty_rows, 0] = True

        x = self.input_projection(state) + self.position_embedding[:, : state.size(1), :]
        attn_out = self._attention(x, attn_valid_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        q_values = self.q_head(x).squeeze(-1)
        return q_values.masked_fill(~original_valid_mask, -1e9)
