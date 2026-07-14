from __future__ import annotations

import argparse
import json
import os
from typing import Any, Union

from .lm_eval_wrapper import RealDataMoELM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--tasks", required=True, help="comma-separated lm-eval task list")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--verbosity", default="INFO")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main() -> None:
    args = parse_args()
    try:
        from lm_eval import evaluator, utils
    except ImportError as exc:
        raise ImportError(
            "Could not import lm_eval. Run this script in the environment where "
            "lm-evaluation-harness is installed."
        ) from exc

    batch_size: Union[int, str]
    batch_size = int(args.batch_size) if str(args.batch_size).isdigit() else args.batch_size
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    model = RealDataMoELM(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer_dir,
        device=args.device,
        batch_size=batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
    )
    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=batch_size,
        device=args.device,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        verbosity=args.verbosity,
    )
    if results is None:
        raise RuntimeError("lm_eval returned no results")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False, default=_json_default)

    print(json.dumps(results.get("results", {}), indent=2, default=_json_default))
    try:
        print(utils.make_table(results))
    except Exception:
        pass


if __name__ == "__main__":
    main()
