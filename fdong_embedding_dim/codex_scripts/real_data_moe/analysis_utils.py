from __future__ import annotations

import glob
import json
import os
from typing import Iterator, List, Tuple

import torch
from transformers import PreTrainedTokenizerFast

from .model import ModelConfig, RealDataMoEForCausalLM


def iter_fixed_documents(data_dir: str, file_pattern: str = "*.txt") -> Iterator[str]:
    """Read from lexicographically last shards, separate from stream startup."""
    paths = sorted(
        glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True), reverse=True
    )
    if not paths:
        raise FileNotFoundError(f"no part-*.txt files found in {data_dir}")
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = line
                if isinstance(value, str) and value.strip():
                    yield value


def fixed_token_sequences(
    data_dir: str,
    tokenizer: PreTrainedTokenizerFast,
    num_sequences: int,
    sequence_length: int,
) -> Tuple[List[torch.Tensor], List[str]]:
    sequences: List[torch.Tensor] = []
    texts: List[str] = []
    eos = tokenizer.eos_token_id
    for text in iter_fixed_documents(data_dir):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if eos is not None:
            token_ids.append(eos)
        if len(token_ids) < sequence_length:
            continue
        sequences.append(torch.tensor(token_ids[:sequence_length], dtype=torch.long))
        texts.append(text)
        if len(sequences) >= num_sequences:
            break
    if len(sequences) < num_sequences:
        raise RuntimeError(
            f"requested {num_sequences} sequences, found only {len(sequences)}"
        )
    return sequences, texts


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[RealDataMoEForCausalLM, int]:
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = ModelConfig(**payload["model_config"])
    model = RealDataMoEForCausalLM(config)
    model.load_state_dict(payload["model"])
    step = int(payload["step"])
    del payload
    model.to(device).eval()
    return model, step
