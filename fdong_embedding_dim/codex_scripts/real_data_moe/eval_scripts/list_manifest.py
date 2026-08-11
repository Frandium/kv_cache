from __future__ import annotations

import argparse

from .common import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--latest-only", action="store_true")
    args = parser.parse_args()
    for row in load_manifest(args.manifest):
        if args.latest_only and not int(row.get("is_latest", 0)):
            continue
        print(f"{row['size']}\t{row['method']}\t{row['checkpoint']}\t{row['step']}")


if __name__ == "__main__":
    main()
