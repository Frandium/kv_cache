from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PatternData:
    sequences: torch.Tensor
    prefix_index: torch.Tensor
    bone_index: torch.Tensor
    token_names: tuple[str, ...]
    train_mask: torch.Tensor
    test_mask: torch.Tensor
    loss_mask: torch.Tensor | None = None
    sequence_lengths: torch.Tensor | None = None
    bone_lengths: torch.Tensor | None = None
    probe_position: int = 0
    target_position: int = 3

    @property
    def vocab_size(self) -> int:
        return len(self.token_names)


def make_cartesian_patterns(
    num_prefixes: int = 8,
    num_bones: int = 8,
    holdout_stride: int = 0,
) -> PatternData:
    """Create [prefix_i, bone_j_0, bone_j_1, suffix_i] for every (i, j).

    MTP=3 is required for the hidden state at prefix_i to receive direct
    supervision from suffix_i.  A non-zero holdout_stride reserves Cartesian
    pairs satisfying (prefix + bone) % holdout_stride == 0 for evaluation.
    Every prefix and every bone must still occur in the training split.
    """
    if num_prefixes < 2 or num_bones < 2:
        raise ValueError("num_prefixes and num_bones must both be at least 2")
    if holdout_stride == 1:
        raise ValueError("holdout_stride=1 would hold out every pair")

    prefix_start = 0
    bone0_start = prefix_start + num_prefixes
    bone1_start = bone0_start + num_bones
    suffix_start = bone1_start + num_bones

    names = (
        tuple(f"P{i}" for i in range(num_prefixes))
        + tuple(f"B{j}_0" for j in range(num_bones))
        + tuple(f"B{j}_1" for j in range(num_bones))
        + tuple(f"S{i}" for i in range(num_prefixes))
    )

    rows: list[list[int]] = []
    prefix_indices: list[int] = []
    bone_indices: list[int] = []
    train: list[bool] = []
    for prefix in range(num_prefixes):
        for bone in range(num_bones):
            rows.append(
                [
                    prefix_start + prefix,
                    bone0_start + bone,
                    bone1_start + bone,
                    suffix_start + prefix,
                ]
            )
            prefix_indices.append(prefix)
            bone_indices.append(bone)
            is_test = holdout_stride > 1 and (prefix + bone) % holdout_stride == 0
            train.append(not is_test)

    train_mask = torch.tensor(train, dtype=torch.bool)
    test_mask = ~train_mask if holdout_stride > 1 else torch.ones_like(train_mask)
    if not train_mask.any():
        raise ValueError("training split is empty")

    prefix_tensor = torch.tensor(prefix_indices)
    bone_tensor = torch.tensor(bone_indices)
    for prefix in range(num_prefixes):
        if not torch.any(train_mask & (prefix_tensor == prefix)):
            raise ValueError(f"prefix {prefix} is absent from training split")
    for bone in range(num_bones):
        if not torch.any(train_mask & (bone_tensor == bone)):
            raise ValueError(f"bone {bone} is absent from training split")

    return PatternData(
        sequences=torch.tensor(rows, dtype=torch.long),
        prefix_index=prefix_tensor,
        bone_index=bone_tensor,
        token_names=names,
        train_mask=train_mask,
        test_mask=test_mask,
    )


def make_variable_lookup_patterns(
    num_prefixes: int = 8,
    num_bones: int = 8,
    min_bone_length: int = 1,
    max_bone_length: int = 4,
    holdout_stride: int = 4,
) -> PatternData:
    """Create padded [P_i, B_j_0, ..., B_j_{L-1}, S_i] for L in a range.

    The suffix offset from the prefix is L+1.  With MTP=3 and L in 1..4, the
    suffix is inside the MTP horizon only for L=1 and L=2.  This removes the
    fixed offset-3 shortcut in the original lookup dataset.
    """
    if num_prefixes < 2 or num_bones < 2:
        raise ValueError("num_prefixes and num_bones must both be at least 2")
    if min_bone_length < 1 or max_bone_length < min_bone_length:
        raise ValueError("bone length range must satisfy 1 <= min <= max")
    if holdout_stride == 1:
        raise ValueError("holdout_stride=1 would hold out every example")

    lengths = list(range(min_bone_length, max_bone_length + 1))
    prefix_start = 0
    bone_start = prefix_start + num_prefixes
    suffix_start = bone_start + num_bones * max_bone_length
    pad_id = suffix_start + num_prefixes
    max_sequence_length = 1 + max_bone_length + 1

    names = (
        tuple(f"P{i}" for i in range(num_prefixes))
        + tuple(f"B{j}_{position}" for j in range(num_bones) for position in range(max_bone_length))
        + tuple(f"S{i}" for i in range(num_prefixes))
        + ("PAD",)
    )

    rows: list[list[int]] = []
    masks: list[list[bool]] = []
    prefix_indices: list[int] = []
    bone_indices: list[int] = []
    sequence_lengths: list[int] = []
    bone_lengths: list[int] = []
    train: list[bool] = []
    for prefix in range(num_prefixes):
        for bone in range(num_bones):
            for bone_length in lengths:
                real_tokens = [prefix_start + prefix]
                real_tokens.extend(
                    bone_start + bone * max_bone_length + position
                    for position in range(bone_length)
                )
                real_tokens.append(suffix_start + prefix)
                pad_count = max_sequence_length - len(real_tokens)
                rows.append(real_tokens + [pad_id] * pad_count)
                masks.append([True] * len(real_tokens) + [False] * pad_count)
                prefix_indices.append(prefix)
                bone_indices.append(bone)
                sequence_lengths.append(len(real_tokens))
                bone_lengths.append(bone_length)
                is_test = (
                    holdout_stride > 1
                    and (prefix + bone + bone_length) % holdout_stride == 0
                )
                train.append(not is_test)

    train_mask = torch.tensor(train, dtype=torch.bool)
    test_mask = ~train_mask if holdout_stride > 1 else torch.ones_like(train_mask)
    if not train_mask.any():
        raise ValueError("training split is empty")

    prefix_tensor = torch.tensor(prefix_indices)
    bone_tensor = torch.tensor(bone_indices)
    bone_length_tensor = torch.tensor(bone_lengths)
    for prefix in range(num_prefixes):
        if not torch.any(train_mask & (prefix_tensor == prefix)):
            raise ValueError(f"prefix {prefix} is absent from training split")
    for bone in range(num_bones):
        if not torch.any(train_mask & (bone_tensor == bone)):
            raise ValueError(f"bone {bone} is absent from training split")
    for bone_length in lengths:
        if not torch.any(train_mask & (bone_length_tensor == bone_length)):
            raise ValueError(f"bone length {bone_length} is absent from training split")

    return PatternData(
        sequences=torch.tensor(rows, dtype=torch.long),
        prefix_index=prefix_tensor,
        bone_index=bone_tensor,
        token_names=names,
        train_mask=train_mask,
        test_mask=test_mask,
        loss_mask=torch.tensor(masks, dtype=torch.bool),
        sequence_lengths=torch.tensor(sequence_lengths),
        bone_lengths=bone_length_tensor,
        probe_position=0,
        target_position=max_sequence_length - 1,
    )


def make_compositional_patterns(
    num_x: int = 8,
    num_y: int = 8,
    num_bones: int = 8,
    test_fraction: float = 0.25,
    split_seed: int = 20260705,
) -> PatternData:
    """Create [X_a, Y_b, bone_j_0, bone_j_1, S_(a+b mod M)].

    Every compositional prefix pair is crossed with every bone.  Entire (a,b)
    pairs, rather than individual prefix-bone rows, are held out.  The split is
    accepted only if every X, Y, and suffix class appears in both train and test.
    The probe reads h(X_a,Y_b), at position 1, and predicts the suffix at offset 3.
    """
    if num_x != num_y:
        raise ValueError("the initial add-mod task requires num_x == num_y")
    if num_x < 4 or num_bones < 2:
        raise ValueError("use at least 4 prefix values and 2 bones")
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")

    modulus = num_x
    generator = torch.Generator().manual_seed(split_seed)
    pairs = [(a, b) for a in range(num_x) for b in range(num_y)]
    test_count = max(1, round(len(pairs) * test_fraction))
    test_pairs: set[tuple[int, int]] | None = None
    all_x, all_y, all_s = set(range(num_x)), set(range(num_y)), set(range(modulus))
    for _ in range(10_000):
        order = torch.randperm(len(pairs), generator=generator).tolist()
        candidate = {pairs[index] for index in order[:test_count]}
        train_candidate = set(pairs) - candidate

        def coverage(items: set[tuple[int, int]]) -> tuple[set[int], set[int], set[int]]:
            return (
                {a for a, _ in items},
                {b for _, b in items},
                {(a + b) % modulus for a, b in items},
            )

        if coverage(candidate) == (all_x, all_y, all_s) and coverage(train_candidate) == (
            all_x,
            all_y,
            all_s,
        ):
            test_pairs = candidate
            break
    if test_pairs is None:
        raise RuntimeError("could not construct a split with full factor and label coverage")

    x_start = 0
    y_start = x_start + num_x
    bone0_start = y_start + num_y
    bone1_start = bone0_start + num_bones
    suffix_start = bone1_start + num_bones
    names = (
        tuple(f"X{a}" for a in range(num_x))
        + tuple(f"Y{b}" for b in range(num_y))
        + tuple(f"B{j}_0" for j in range(num_bones))
        + tuple(f"B{j}_1" for j in range(num_bones))
        + tuple(f"S{s}" for s in range(modulus))
    )

    rows: list[list[int]] = []
    composition_indices: list[int] = []
    bone_indices: list[int] = []
    train: list[bool] = []
    for a, b in pairs:
        suffix = (a + b) % modulus
        for bone in range(num_bones):
            rows.append(
                [
                    x_start + a,
                    y_start + b,
                    bone0_start + bone,
                    bone1_start + bone,
                    suffix_start + suffix,
                ]
            )
            composition_indices.append(a * num_y + b)
            bone_indices.append(bone)
            train.append((a, b) not in test_pairs)

    train_mask = torch.tensor(train, dtype=torch.bool)
    return PatternData(
        sequences=torch.tensor(rows, dtype=torch.long),
        prefix_index=torch.tensor(composition_indices),
        bone_index=torch.tensor(bone_indices),
        token_names=names,
        train_mask=train_mask,
        test_mask=~train_mask,
        probe_position=1,
        target_position=4,
    )
