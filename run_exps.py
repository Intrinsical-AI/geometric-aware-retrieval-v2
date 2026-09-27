#!/usr/bin/env python3
"""FiQA-first matrix runner for the BEIR benchmark pack."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "fiqa"
DEFAULT_DOC_SIZES = (1000, 5000)
DEFAULT_PPR_TOPKS = (100, 200)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the fixed-matrix runner parser."""
    parser = argparse.ArgumentParser(
        description="Run the fixed FiQA benchmark matrix for hard vs soft graph retrieval."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="BEIR dataset name.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Explicit local BEIR dataset directory. Used when it exists.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory used for downloaded BEIR datasets.",
    )
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--allow-download",
        dest="allow_download",
        action="store_true",
        help="Download the dataset when no local dataset is available.",
    )
    download_group.add_argument(
        "--no-download",
        dest="allow_download",
        action="store_false",
        help="Fail fast instead of downloading when the local dataset is missing.",
    )
    parser.set_defaults(allow_download=False)
    parser.add_argument("--max-queries", type=int, default=100, help="Fixed query budget per run.")
    parser.add_argument("--k", type=int, default=20, help="Target graph degree.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size for the benchmark.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for all runs.")
    return parser


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    """Build the fixed FiQA benchmark matrix."""
    script = Path("research") / "beir_euclidean_vs_geo.py"
    common_args = [
        sys.executable,
        str(script),
        "--dataset",
        args.dataset,
        "--download-dir",
        str(args.download_dir),
        "--max-queries",
        str(args.max_queries),
        "--k",
        str(args.k),
        "--batch-size",
        str(args.batch_size),
        "--device",
        "cpu",
        "--seed",
        str(args.seed),
        "--allow-download" if args.allow_download else "--no-download",
    ]
    if args.dataset_dir is not None:
        common_args.extend(["--dataset-dir", str(args.dataset_dir)])

    commands: list[list[str]] = []
    for max_docs in DEFAULT_DOC_SIZES:
        commands.append(common_args + ["--max-docs", str(max_docs), "--rerank", "none"])
        for ppr_topk in DEFAULT_PPR_TOPKS:
            commands.append(
                common_args
                + [
                    "--max-docs",
                    str(max_docs),
                    "--rerank",
                    "ppr",
                    "--ppr-topk",
                    str(ppr_topk),
                ]
            )
    return commands


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    commands = build_commands(args)
    for command in commands:
        print("Ejecutando:", " ".join(command))
        subprocess.run(command, check=True)
    print("\n✅ Matriz FiQA completada.")


if __name__ == "__main__":
    main()
