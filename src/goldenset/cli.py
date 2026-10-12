"""The `goldenset` command line. Subcommands are added by later tasks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _add_generate(sub: Any) -> None:
    parser = sub.add_parser("generate", help="Generate a synthetic trace dataset with ground truth")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(func=_cmd_generate)


def _cmd_generate(args: argparse.Namespace) -> int:
    from goldenset.generator.build import generate_dataset, write_dataset

    dataset = generate_dataset(args.n, args.seed)
    write_dataset(dataset, args.out)
    print(f"Wrote {args.n} traces to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="goldenset", description="Golden-set pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_generate(sub)
    # --- add new subcommands above this line ---
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
