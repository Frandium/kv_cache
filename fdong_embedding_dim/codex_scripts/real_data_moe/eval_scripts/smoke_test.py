from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .common import (
    forward_flops_per_token,
    job_fingerprint,
    parse_flop_targets,
    read_csv,
    training_flops,
    write_csv,
)
from .predictability_eval import (
    LayerPairLinearPredictor,
    accumulate_accuracy,
    predictor_loss,
)
from .routing_eval import lru_counts
from .swap_latency_eval import SwappingDecodeRuntime
try:
    from moe.model import ModelConfig, RealDataMoEForCausalLM
except ModuleNotFoundError:
    from fdong_embedding_dim.codex_scripts.real_data_moe.model import (
        ModelConfig,
        RealDataMoEForCausalLM,
    )


def test_lru() -> None:
    routes = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 0]])
    loads, evictions = lru_counts(routes, [1, 2, 4])
    assert loads[:, 0].tolist() == [4, 2, 2]
    assert loads[:, 1].tolist() == [4, 4, 3]
    assert torch.all(loads[1:] <= loads[:-1])
    assert torch.all(evictions[1:] <= evictions[:-1])


def test_predictor() -> None:
    torch.manual_seed(1)
    layers, batch, tokens, hidden_size, experts = 3, 2, 5, 7, 8
    predictor = LayerPairLinearPredictor(layers, hidden_size, experts)
    hidden = [torch.randn(batch, tokens, hidden_size) for _ in range(layers)]
    routes = torch.randint(0, experts, (layers, batch, tokens))
    loss = predictor_loss(predictor, hidden, routes)
    loss.backward()
    assert all(parameter.grad is not None for parameter in predictor.parameters())
    correct = torch.zeros(2, layers, layers, 3, dtype=torch.long)
    totals = torch.zeros(2, layers, layers, dtype=torch.long)
    accumulate_accuracy(predictor, hidden, routes, correct, totals)
    assert int((totals[0] > 0).sum()) == 3  # strict upper triangle
    assert int((totals[1] > 0).sum()) == 6  # diagonal and lower triangle
    assert torch.all(correct[..., 0] <= correct[..., 1])
    assert torch.all(correct[..., 1] <= correct[..., 2])


def test_flops_and_csv() -> None:
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "common_intermediate_size": 4,
        "tail_intermediate_size": 8,
        "num_tail_experts": 8,
        "vocab_size": 32,
    }
    train_args = {
        "sequence_length": 16,
        "batch_size": 2,
        "gradient_accumulation": 3,
    }
    flops = forward_flops_per_token(config, train_args)
    assert all(value > 0 for value in flops.values())
    totals = training_flops(
        {"model_config": config, "train_args": train_args, "world_size": 4}, 10
    )
    assert totals["training_tokens"] == 3840
    assert parse_flop_targets("2e3,1e3,2e3") == [1000.0, 2000.0]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.csv"
        write_csv(path, [{"a": 1, "b": "x"}])
        assert read_csv(path) == [{"a": "1", "b": "x"}]
        assert job_fingerprint(path, "v1", "x=1") == job_fingerprint(path, "v1", "x=1")


def test_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(3)
    config = ModelConfig.proposed(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        num_tail_experts=4,
        num_experts_per_token=1,
        common_intermediate_size=4,
        tail_intermediate_size=8,
        router_window=3,
        orthogonalize_tail=True,
        orthogonal_rank=4,
    )
    reference = RealDataMoEForCausalLM(config).eval().to(dtype=torch.bfloat16)
    tokens = torch.tensor([[1, 7, 5, 9, 2]], dtype=torch.long)
    with torch.inference_mode():
        full_logits, _ = reference(tokens)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save(
            {"step": 10, "model_config": reference.config_dict(), "model": reference.state_dict()},
            checkpoint,
        )
        runtime = SwappingDecodeRuntime(str(checkpoint), 4, torch.device("cpu"), 16)
        incremental = []
        with torch.inference_mode():
            for token in tokens[0]:
                incremental.append(runtime.token_step(token))
        incremental_logits = torch.cat(incremental, dim=1)
        runtime.reset()
        with torch.inference_mode():
            prompt_logits = runtime.prompt_forward(tokens[0])
    assert torch.allclose(full_logits.float(), incremental_logits.float(), atol=0.08, rtol=0.03)
    assert torch.allclose(full_logits[:, -1:].float(), prompt_logits.float(), atol=0.08, rtol=0.03)


def main() -> None:
    test_lru()
    test_predictor()
    test_flops_and_csv()
    test_cached_decode_matches_full_forward()
    print("evaluation suite smoke tests passed")


if __name__ == "__main__":
    main()
