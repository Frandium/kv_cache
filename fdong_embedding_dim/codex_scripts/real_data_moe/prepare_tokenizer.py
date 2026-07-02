from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def documents(data_dir: str, max_documents: int) -> Iterator[str]:
    count = 0
    for path in sorted(glob.glob(os.path.join(data_dir, "part-*.txt"))):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, str) and value.strip():
                    yield value
                    count += 1
                    if count >= max_documents:
                        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--max-documents", type=int, default=200_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tokenizer_json = os.path.join(args.output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json) and not args.force:
        print(f"[tokenizer] reuse {args.output_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        documents(args.data_dir, args.max_documents), trainer=trainer
    )
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    wrapped.save_pretrained(args.output_dir)
    if len(wrapped) != args.vocab_size:
        raise RuntimeError(f"expected vocab {args.vocab_size}, got {len(wrapped)}")
    print(f"[tokenizer] saved vocab={len(wrapped)} to {args.output_dir}")


if __name__ == "__main__":
    main()
