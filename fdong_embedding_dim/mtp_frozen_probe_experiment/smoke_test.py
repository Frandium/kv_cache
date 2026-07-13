from __future__ import annotations

import torch

from .data import make_cartesian_patterns, make_compositional_patterns
from .experiment import (
    evaluate_probe,
    multi_token_loss,
    train_frozen_probe,
    train_model,
)


def main() -> None:
    data = make_cartesian_patterns(num_prefixes=4, num_bones=4, holdout_stride=3)
    assert data.sequences.shape == (16, 4)
    assert data.sequences[0, 0] == data.sequences[1, 0]
    assert data.sequences[0, 3] == data.sequences[1, 3]
    assert data.sequences[0, 1] != data.sequences[1, 1]

    for backbone in ("linear", "mlp", "attention"):
        model, _ = train_model(
            data=data,
            backbone_kind=backbone,
            hidden_size=4,
            mtp=3,
            steps=5,
            learning_rate=1e-2,
            seed=1,
            device=torch.device("cpu"),
        )
        hidden, logits = model(data.sequences)
        assert hidden.shape == (16, 4, 4)
        assert len(logits) == 3
        assert torch.isfinite(multi_token_loss(logits, data.sequences))
        probe, _ = train_frozen_probe(
            model=model,
            data=data,
            probe_kind="linear",
            steps=5,
            learning_rate=1e-2,
            seed=2,
            device=torch.device("cpu"),
        )
        metrics = evaluate_probe(
            model,
            probe,
            data.sequences[data.test_mask],
            data.probe_position,
            data.target_position,
        )
        assert 0.0 <= metrics["accuracy"] <= 1.0

    compositional = make_compositional_patterns(
        num_x=8, num_y=8, num_bones=3, test_fraction=0.25
    )
    assert compositional.sequences.shape == (8 * 8 * 3, 5)
    assert compositional.probe_position == 1
    assert compositional.target_position == 4
    train_pairs = set(compositional.prefix_index[compositional.train_mask].tolist())
    test_pairs = set(compositional.prefix_index[compositional.test_mask].tolist())
    assert train_pairs.isdisjoint(test_pairs)
    print("smoke test passed")


if __name__ == "__main__":
    main()
