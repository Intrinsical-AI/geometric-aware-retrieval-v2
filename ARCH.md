# Project Architecture

This document provides a high level overview of the geoIR code base and the optional
extras listed in `pyproject.toml`.

## Layout

```
geoIR/
  core/        # experiment helpers and registry
  data/        # lightweight data loading
  eval/        # evaluation metrics
  geo/         # geometry utilities and differentiable components
  retrieval/   # encoder, index and high level retriever API
  training/    # training utilities
```

The library is designed around a small registry located in `geoIR/core/registry.py`.
Modules register default implementations under groups such as `encoder` so that
higher level helpers (e.g. `geoIR.load_encoder`) can load them without direct imports.

## Optional Extras

Additional dependencies are grouped into extras:

- **viz** – plotting libraries for graph visualisation.
- **hf** – HuggingFace ecosystem packages used for most examples.
- **rust** – reserved for future compiled extensions.
- **dev** – tooling for development (tests, linting, type checking).

To install with selected extras use:

```bash
pip install -e ".[dev,hf,viz]"
```

## Experiments

Training and evaluation utilities are located in `geoIR.training` and `geoIR.eval`.
The `quick_experiment()` helper in `geoIR.__init__` demonstrates a minimal workflow
and can be used as a starting point for custom experiments.
