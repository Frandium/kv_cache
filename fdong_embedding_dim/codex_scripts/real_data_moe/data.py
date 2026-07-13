from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerFast


@dataclass
class StreamState:
    epoch: int = 0
    file_index: int = 0
    byte_offset: int = 0
    token_buffer: List[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.token_buffer is None:
            self.token_buffer = []


class DCLMTokenStream:
    """Deterministic, stateful DCLM stream with byte-exact resume support."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizerFast,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        file_pattern: str = "*.txt",
    ) -> None:
        all_files = sorted(
            glob.glob(os.path.join(data_dir, "**", file_pattern), recursive=True)
        )
        self.files = all_files[rank::world_size]
        if not self.files:
            raise FileNotFoundError(
                f"no {file_pattern} files assigned to rank {rank}/{world_size} in {data_dir}"
            )
        self.tokenizer = tokenizer
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.state = StreamState()
        self._handle = None
        self._order: List[str] = []
        self._set_epoch(0)

    def _set_epoch(self, epoch: int) -> None:
        self.state.epoch = epoch
        self._order = list(self.files)
        random.Random(self.seed + epoch).shuffle(self._order)

    def _open_current_file(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = open(self._order[self.state.file_index], "r", encoding="utf-8")
        self._handle.seek(self.state.byte_offset)

    def _advance_file(self) -> None:
        self.state.file_index += 1
        self.state.byte_offset = 0
        if self.state.file_index >= len(self._order):
            self.state.file_index = 0
            self._set_epoch(self.state.epoch + 1)
        self._open_current_file()

    def _read_document(self) -> str:
        if self._handle is None:
            self._open_current_file()
        while True:
            line = self._handle.readline()
            if line:
                self.state.byte_offset = self._handle.tell()
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = line
                if isinstance(value, str) and value.strip():
                    return value
            else:
                self._advance_file()

    def next_batch(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needed = batch_size * (sequence_length + 1)
        eos = self.tokenizer.eos_token_id
        while len(self.state.token_buffer) < needed:
            encoded = self.tokenizer.encode(self._read_document(), add_special_tokens=False)
            self.state.token_buffer.extend(encoded)
            if eos is not None:
                self.state.token_buffer.append(eos)
        tokens = self.state.token_buffer[:needed]
        del self.state.token_buffer[:needed]
        tensor = torch.tensor(tokens, dtype=torch.long).view(batch_size, sequence_length + 1)
        return tensor[:, :-1].to(device), tensor[:, 1:].to(device)

    def state_dict(self) -> Dict[str, object]:
        return {
            "epoch": self.state.epoch,
            "file_index": self.state.file_index,
            "byte_offset": self.state.byte_offset,
            "token_buffer": list(self.state.token_buffer),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.state = StreamState(
            epoch=int(state["epoch"]),
            file_index=int(state["file_index"]),
            byte_offset=int(state["byte_offset"]),
            token_buffer=list(state["token_buffer"]),  # type: ignore[arg-type]
        )
        self._set_epoch(self.state.epoch)
        self._open_current_file()
