from __future__ import annotations

import argparse
import tempfile

import torch

from .model import ModelConfig, RealDataMoEForCausalLM, parameter_counts


def tiny_config(orthogonalize_tail: bool) -> ModelConfig:
    return ModelConfig.proposed(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        common_intermediate_size=16,
        tail_intermediate_size=36,
        orthogonalize_tail=orthogonalize_tail,
        orthogonal_rank=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    baseline = RealDataMoEForCausalLM(ModelConfig.baseline())
    proposed = RealDataMoEForCausalLM(ModelConfig.proposed())
    baseline_count = parameter_counts(baseline)["total"]
    proposed_count = parameter_counts(proposed)["total"]
    assert baseline_count == proposed_count == 57_429_248
    del baseline, proposed

    config = tiny_config(orthogonalize_tail=True)
    model = RealDataMoEForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    logits, diagnostics = model(input_ids)
    auxiliary_loss = torch.stack(
        [layer["load_balance_loss"] for layer in diagnostics.values()]
    ).mean()
    assert auxiliary_loss.requires_grad
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    (loss + 0.01 * auxiliary_loss).backward()
    optimizer.step()
    assert model.layers[0].moe.router.weight.grad is not None
    assert any(
        expert.down_proj.weight.grad is not None
        for expert in model.layers[0].moe.tail_experts
    )

    moe = model.layers[0].moe
    raw = torch.randn(13, config.hidden_size, device=device, requires_grad=True)
    projected = moe._orthogonalize(raw)
    leakage = (projected @ moe._common_basis).abs().max().item()
    assert leakage < 1e-5, leakage
    projected.square().mean().backward()
    assert raw.grad is not None

    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 1},
            checkpoint.name,
        )
        restored = RealDataMoEForCausalLM(config)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        state = torch.load(checkpoint.name, map_location="cpu", weights_only=False)
        restored.load_state_dict(state["model"])
        restored_optimizer.load_state_dict(state["optimizer"])
        assert state["step"] == 1

    print(
        f"smoke test passed: params={baseline_count:,}, "
        f"loss={loss.detach().item():.4f}, orthogonal_leakage={leakage:.3e}"
    )

    checkpoint_config = tiny_config(orthogonalize_tail=False)
    checkpoint_config.gradient_checkpointing = True
    checkpoint_model = RealDataMoEForCausalLM(checkpoint_config)
    checkpoint_ids = torch.randint(0, checkpoint_config.vocab_size, (2, 16))
    checkpoint_logits, _ = checkpoint_model(checkpoint_ids)
    checkpoint_loss = torch.nn.functional.cross_entropy(
        checkpoint_logits[:, :-1].reshape(-1, checkpoint_config.vocab_size),
        checkpoint_ids[:, 1:].reshape(-1),
    )
    checkpoint_loss.backward()
    assert checkpoint_model.layers[0].moe.common_expert.down_proj.weight.grad is not None
    print("gradient-checkpoint forward/backward passed")


if __name__ == "__main__":
    main()
