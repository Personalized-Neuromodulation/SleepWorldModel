from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import build_hsp_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HSP HDF5 JSONL manifest")
    parser.add_argument("--root", type=Path, required=True, help="HSP cohort root")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Subject split seed")
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional smoke-test limit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def report_progress(count: int, path: Path, status: str) -> None:
        if count == 1 or count % 100 == 0 or status != "ok":
            print(f"[{count}] {status}: {path}", flush=True)

    summary = build_hsp_manifest(
        args.root,
        args.output,
        split_seed=args.seed,
        split_ratios=tuple(args.split_ratios),
        limit=args.limit,
        progress_callback=report_progress,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
