from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import load_manifest


def copy_file(source: Path, target: Path) -> None:
    if source.is_file() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"[migrate] {source} -> {target}", flush=True)


def copy_tree(source: Path, target: Path) -> None:
    if source.is_dir() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        print(f"[migrate] {source} -> {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-dir", required=True)
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    manifest = load_manifest(args.manifest)
    legacy_loss = raw / "test_loss.json"
    old_loss = json.loads(legacy_loss.read_text()) if legacy_loss.is_file() else {}
    for run in manifest:
        size, method, step = run["size"], run["method"], int(run["step"])
        label = f"{size}_{method}"
        copy_tree(
            raw / "routing" / label,
            raw / "routing" / "v1" / size / method / f"step{step:07d}",
        )
        copy_tree(
            raw / "predictability" / label,
            raw / "predictability" / "v1" / size / method / f"step{step:07d}",
        )
        for capacity in (1, 2, 4, 8):
            copy_file(
                raw / "swap_latency" / f"{label}_k{capacity}.csv",
                raw / "swap_latency" / "decode_v1" / size / method
                / f"step{step:07d}" / f"k{capacity}.csv",
            )
        copy_file(
            raw / "lm_eval" / f"{label}.json",
            raw / "scaling" / "lm_eval_v1" / size / method
            / f"step{step:07d}" / "result.json",
        )
        if label in old_loss:
            target = (
                raw / "scaling" / "test_loss_v1" / size / method
                / f"step{step:07d}" / "result.json"
            )
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps({f"{label}_step{step}": old_loss[label]}, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"[migrate] {legacy_loss}:{label} -> {target}", flush=True)


if __name__ == "__main__":
    main()
