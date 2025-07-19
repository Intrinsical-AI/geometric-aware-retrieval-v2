#!/usr/bin/env python3
"""Minimal command-line wrapper around geoIR.quick_experiment.

Example:
    python examples/quick_experiment_cli.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --dataset beir/fiqa \
        --k 30

The script prints the resulting metrics as prettified JSON.
"""
from __future__ import annotations

import argparse
import json
import geoIR as gi


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Run a quick geometric retrieval experiment")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder model name or path")
    parser.add_argument("--dataset", default="beir/fiqa", help="BEIR dataset name or local path")
    parser.add_argument("--k", type=int, default=20, help="k for k-NN graph")
    parser.add_argument("--device", default=None, help="Force device (cpu, cuda, mps)")
    parser.add_argument("--no-geo", action="store_true", help="Disable geometric regularisation (baseline)")
    args = parser.parse_args()

    results = gi.quick_experiment(
        model_name=args.model,
        dataset=args.dataset,
        k=args.k,
        geometric=not args.no_geo,
        device=args.device,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
