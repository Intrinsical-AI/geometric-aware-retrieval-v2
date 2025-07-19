# Project Architecture

This document gives a brief overview of the repository structure and the optional extras defined in `pyproject.toml`.

## Layout

- `geoIR/` – Library code organised in modules:
  - `core/` contains configuration helpers and a simple experiment runner.
  - `data/` provides dataset loading utilities.
  - `eval/` hosts evaluation metrics.
  - `geo/` implements geometric primitives such as the differentiable soft-kNN.
  - `retrieval/` exposes the high level API (`GeometricRetriever`, `Encoder`).
  - `training/` includes the generic `Trainer` used by `quick_experiment`.
- `examples/` – Small runnable scripts showcasing common usage patterns.
- `tests/` – Unit tests for core components.

## Optional Dependencies

The library defines several optional extras that extend its functionality:

- `viz` – visualisation packages like `plotly` and `umap-learn`.
- `hf` – HuggingFace ecosystem packages for dataset loading and model fine-tuning.
- `rust` – reserved for future compiled extensions.
- `dev` – development tools such as `pytest`, `ruff` and `mypy`.

Install them with `pip install -e .[viz,hf]` for example.  Refer to the README for more installation notes.
