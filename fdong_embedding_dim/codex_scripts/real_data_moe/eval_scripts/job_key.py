from __future__ import annotations

import argparse

from .common import job_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(job_fingerprint(args.checkpoint, args.protocol, args.config))


if __name__ == "__main__":
    main()
