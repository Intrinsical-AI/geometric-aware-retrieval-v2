# Architecture Overview

This document describes the overall structure of the **geoIR** project and the optional extras that can be installed via `pip`.

## Package Layout

```
geoIR/
  core/        # configuration and experiment utilities
  retrieval/   # encoder and index abstractions
  geo/         # geometric operations (graphs, curvature, differentiable pipeline)
  eval/        # metrics and LLM-based judges
  training/    # trainer class for fine-tuning
```

The package is organised around small, composable modules. The central registry in `geoIR.core.registry` exposes factories for encoders and other components.

## Optional Dependencies

Several extras enable additional functionality:

- **`viz`** – plotting utilities (`plotly`, `dash`, `umap-learn`, `matplotlib`, `seaborn`).
- **`hf`** – integration with the HuggingFace ecosystem (`transformers`, `datasets`, `sentence-transformers`).
- **`dev`** – development tools (`pytest`, `ruff`, `black`, `mypy`, `ipykernel`, `beir`).

Install any of them with `pip install "geoIR[extra]"` (for example `geoIR[viz]`).

## Data Flow

A typical experiment follows these steps:

1. **Encoder Loading** – `geoIR.load_encoder` retrieves a HuggingFace model and wraps it in the `Encoder` class.
2. **Index Building** – `Encoder.build_index` encodes a corpus and constructs a k‑NN graph.
3. **Training** – the `Trainer` class optimises the encoder using differentiable geometric loss functions.
4. **Evaluation** – metrics in `geoIR.eval` compute classic and geometry‑aware retrieval scores.

See `README.md` for concrete examples.
