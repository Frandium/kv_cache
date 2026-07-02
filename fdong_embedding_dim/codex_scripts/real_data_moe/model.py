from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 16_384
    hidden_size: int = 768
    num_hidden_layers: int = 4
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    max_position_embeddings: int = 1_024
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    num_tail_experts: int = 4
    common_intermediate_size: int = 384
    tail_intermediate_size: int = 864
    router_input: str = "attention_mean"
    router_window: int = 16
    orthogonalize_tail: bool = False
    orthogonal_rank: int = 16
    orthogonal_refresh_steps: int = 50
    gradient_checkpointing: bool = False

    @classmethod
    def baseline(cls, **overrides: object) -> "ModelConfig":
        values: Dict[str, object] = {
            "common_intermediate_size": 768,
            "tail_intermediate_size": 768,
            "router_input": "residual",
            "orthogonalize_tail": False,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def proposed(cls, **overrides: object) -> "ModelConfig":
        values: Dict[str, object] = {
            "common_intermediate_size": 384,
            "tail_intermediate_size": 864,
            "router_input": "attention_mean",
        }
        values.update(overrides)
        return cls(**values)

    def validate(self) -> None:
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_attention_heads * head_dim must equal hidden_size")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.router_input not in {"residual", "attention", "attention_mean"}:
            raise ValueError("router_input must be residual, attention, or attention_mean")
        if self.router_window < 1:
            raise ValueError("router_window must be positive")
        if not 0 < self.orthogonal_rank <= min(
            self.hidden_size, self.common_intermediate_size
        ):
            raise ValueError("orthogonal_rank exceeds common down-projection rank")
        if self.orthogonal_refresh_steps < 1:
            raise ValueError("orthogonal_refresh_steps must be positive")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(x.dtype)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_std: float) -> None:
        super().__init__(in_features, out_features, bias=False)
        nn.init.normal_(self.weight, mean=0.0, std=init_std)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_positions).float()
        angles = torch.outer(positions, inv_freq)
        embedding = torch.repeat_interleave(angles, repeats=2, dim=-1)
        self.register_buffer("cos", embedding.cos(), persistent=False)
        self.register_buffer("sin", embedding.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos = self.cos[:seq_len].to(device=q.device, dtype=q.dtype)[None, None]
        sin = self.sin[:seq_len].to(device=q.device, dtype=q.dtype)[None, None]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class QwenAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        self.config = config
        self.q_proj = Linear(config.hidden_size, q_dim, config.initializer_range)
        self.k_proj = Linear(config.hidden_size, kv_dim, config.initializer_range)
        self.v_proj = Linear(config.hidden_size, kv_dim, config.initializer_range)
        self.o_proj = Linear(q_dim, config.hidden_size, config.initializer_range)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rope(q, k)
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class SwiGLUExpert(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = Linear(
            config.hidden_size, intermediate_size, config.initializer_range
        )
        self.up_proj = Linear(
            config.hidden_size, intermediate_size, config.initializer_range
        )
        self.down_proj = Linear(
            intermediate_size, config.hidden_size, config.initializer_range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def causal_window_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Causal moving mean, including the current attention output."""
    if window == 1:
        return x
    prefix = torch.cat(
        (torch.zeros_like(x[:, :1], dtype=torch.float32), torch.cumsum(x.float(), dim=1)),
        dim=1,
    )
    positions = torch.arange(x.shape[1], device=x.device)
    starts = (positions + 1 - window).clamp_min(0)
    totals = prefix[:, positions + 1] - prefix[:, starts]
    counts = (positions + 1 - starts).view(1, -1, 1)
    return (totals / counts).to(x.dtype)


class CommonTailMoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.common_expert = SwiGLUExpert(config, config.common_intermediate_size)
        self.tail_experts = nn.ModuleList(
            SwiGLUExpert(config, config.tail_intermediate_size)
            for _ in range(config.num_tail_experts)
        )
        self.router = Linear(
            config.hidden_size, config.num_tail_experts, config.initializer_range
        )
        self.register_buffer("_common_basis", torch.empty(0), persistent=False)
        self._basis_step = -1
        self.training_step = 0
        self.last_route_counts: Optional[torch.Tensor] = None

    def set_training_step(self, step: int) -> None:
        self.training_step = step

    @torch.no_grad()
    def _refresh_common_basis(self, device: torch.device, dtype: torch.dtype) -> None:
        # CPU float32 SVD avoids MPS float64/backward limitations. The detached
        # basis is a constraint target, not a differentiable decomposition.
        weight = self.common_expert.down_proj.weight.detach().float().cpu()
        left, _, _ = torch.linalg.svd(weight, full_matrices=False)
        self._common_basis = left[:, : self.config.orthogonal_rank].to(
            device=device, dtype=dtype
        )
        self._basis_step = self.training_step

    def _basis(self, output: torch.Tensor) -> torch.Tensor:
        stale = (
            self._common_basis.numel() == 0
            or self._common_basis.device != output.device
            or self._common_basis.dtype != output.dtype
            or self.training_step - self._basis_step >= self.config.orthogonal_refresh_steps
        )
        if stale:
            self._refresh_common_basis(output.device, output.dtype)
        return self._common_basis.detach()

    def _orthogonalize(self, output: torch.Tensor) -> torch.Tensor:
        basis = self._basis(output)
        return output - (output @ basis) @ basis.transpose(0, 1)

    def forward(
        self,
        x: torch.Tensor,
        attention_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.config.router_input == "residual":
            router_input = x
        elif self.config.router_input == "attention":
            router_input = attention_output
        else:
            router_input = causal_window_mean(attention_output, self.config.router_window)

        router_logits = self.router(router_input)
        router_probs = F.softmax(router_logits.float(), dim=-1).to(x.dtype)
        routes = router_logits.argmax(dim=-1)
        selected_prob = router_probs.gather(-1, routes.unsqueeze(-1)).squeeze(-1)

        tail_output = torch.zeros_like(x)
        counts = torch.zeros(
            self.config.num_tail_experts, device=x.device, dtype=torch.long
        )
        for expert_index, expert in enumerate(self.tail_experts):
            mask = routes == expert_index
            counts[expert_index] = mask.sum()
            if mask.any():
                expert_output = expert(x[mask])
                if self.config.orthogonalize_tail:
                    expert_output = self._orthogonalize(expert_output)
                # Forward scale is exactly one; backward supplies a sparse
                # straight-through gradient to the selected router probability.
                probability = selected_prob[mask]
                gate = probability / probability.detach().clamp_min(1e-6)
                tail_output[mask] = expert_output * gate.unsqueeze(-1)

        self.last_route_counts = counts.detach().cpu()
        diagnostics = {
            "route_counts": counts.detach(),
            "router_entropy": (
                -(router_probs.float() * router_probs.float().clamp_min(1e-9).log())
                .sum(dim=-1)
                .mean()
                .detach()
            ),
        }
        return self.common_expert(x) + tail_output, diagnostics


class QwenMoEBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = QwenAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = CommonTailMoE(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attention_output = self.self_attn(self.input_layernorm(x))
        x = x + attention_output
        moe_output, diagnostics = self.moe(
            self.post_attention_layernorm(x), attention_output
        )
        return x + moe_output, diagnostics


class RealDataMoEForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=config.initializer_range)
        self.layers = nn.ModuleList(QwenMoEBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def set_training_step(self, step: int) -> None:
        for layer in self.layers:
            layer.moe.set_training_step(step)

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        x = self.embed_tokens(input_ids)
        diagnostics: Dict[str, Dict[str, torch.Tensor]] = {}
        for index, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(lambda value: layer(value)[0], x, use_reentrant=False)
            else:
                x, layer_diagnostics = layer(x)
                diagnostics[f"layer_{index}"] = layer_diagnostics
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight), diagnostics

    def config_dict(self) -> Dict[str, object]:
        return asdict(self.config)


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    groups = {"embedding": 0, "attention": 0, "common": 0, "tail": 0, "router_norm": 0}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith("embed_tokens"):
            groups["embedding"] += count
        elif ".self_attn." in name:
            groups["attention"] += count
        elif ".common_expert." in name:
            groups["common"] += count
        elif ".tail_experts." in name:
            groups["tail"] += count
        else:
            groups["router_norm"] += count
    groups["total"] = sum(groups.values())
    return groups
