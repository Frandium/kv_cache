from __future__ import annotations

import torch
from torch import nn


class CausalBackbone(nn.Module):
    """Tiny causal backbone with linear, MLP, or single-attention computation.

    Linear and MLP variants first take a cumulative mean of token embeddings.
    Without this causal aggregation, the state at B_j_1 could not read P_i and
    the model could not solve the ordinary next-token suffix prediction task.
    The attention variant is intentionally minimal: token embeddings plus
    position embeddings followed by one causal QKV self-attention layer and no
    feed-forward network.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        kind: str,
        num_attention_heads: int = 1,
        max_seq_len: int = 4,
    ) -> None:
        super().__init__()
        if kind not in {"linear", "mlp", "attention"}:
            raise ValueError(f"unknown backbone kind: {kind}")
        if kind == "attention" and hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.kind = kind
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if kind == "linear":
            self.body = nn.Linear(hidden_size, hidden_size)
        elif kind == "mlp":
            self.body = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),
                nn.GELU(),
                nn.Linear(2 * hidden_size, hidden_size),
            )
        else:
            self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
            self.attention = nn.MultiheadAttention(
                hidden_size, num_attention_heads, batch_first=True
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        if self.kind in {"linear", "mlp"}:
            steps = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype)
            causal_mean = x.cumsum(dim=1) / steps.view(1, -1, 1)
            return self.body(causal_mean)

        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        seq_len = token_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_output, _ = self.attention(
            x, x, x, attn_mask=causal_mask, need_weights=False
        )
        return attn_output


class MultiTokenModel(nn.Module):
    """Shared causal backbone with independent linear heads for offsets 1..K."""

    def __init__(self, backbone: CausalBackbone, vocab_size: int, mtp: int) -> None:
        super().__init__()
        if mtp < 1:
            raise ValueError("mtp must be >= 1; mtp=1 is NTP")
        self.backbone = backbone
        self.mtp = mtp
        self.heads = nn.ModuleList(
            nn.Linear(backbone.hidden_size, vocab_size) for _ in range(mtp)
        )

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = self.backbone(token_ids)
        return hidden, [head(hidden) for head in self.heads]


class Probe(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(hidden_size, vocab_size)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),
                nn.GELU(),
                nn.Linear(2 * hidden_size, vocab_size),
            )
        else:
            raise ValueError(f"unknown probe kind: {kind}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
