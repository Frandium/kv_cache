from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--required-gpus", type=int, default=8)
    args = parser.parse_args()

    if not Path(args.data_dir).is_dir():
        raise FileNotFoundError(f"data directory not found: {args.data_dir}")
    if not Path(args.tokenizer_dir).is_dir():
        raise FileNotFoundError(f"tokenizer directory not found: {args.tokenizer_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA/PPU backend is unavailable")
    count = torch.cuda.device_count()
    if count < args.required_gpus:
        raise RuntimeError(f"requires {args.required_gpus} devices, found {count}")
    import transformers  # noqa: F401
    import lm_eval  # noqa: F401

    for index in range(args.required_gpus):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"[preflight] device={index} name={properties.name} "
            f"memory_gib={properties.total_memory / 2**30:.2f}",
            flush=True,
        )
    print("[preflight] dependencies and paths passed", flush=True)


if __name__ == "__main__":
    main()
