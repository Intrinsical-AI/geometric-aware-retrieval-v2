# Geometric-Aware Retrieval with Differentiable Soft-kNN

A Python library for geometric-aware information retrieval using differentiable k-nearest neighbors and graph-based re-ranking.

## Features

- **Geometric-Aware Retrieval**: Utilizes graph-based distances to capture semantic relationships, leading to more meaningful search results.
- **Differentiable Pipeline**: The entire retrieval process is end-to-end trainable, allowing for fine-tuning and optimization.
- **Multiple Backends**: Supports FAISS for efficient similarity search, with the flexibility to integrate other backends.
- **Pre-trained Models**: Seamlessly integrates with HuggingFace's `sentence-transformers` for access to a wide range of pre-trained models.
- **Comprehensive Evaluation**: Includes built-in support for standard information retrieval metrics (e.g., MAP, NDCG) and novel geometric metrics (e.g., RARE, SUD).
- **Modular Design**: The library is organized into distinct modules for easy extension and customization.

## Modules

### `geoIR.core`

The core module provides the foundational components of the library, including configuration management and the main runner for experiments.

- `config.py`: Defines the data structures for configuring experiments using Pydantic.
- `runner.py`: The main entry point for running experiments.

### `geoIR.data`

The data module handles loading and preprocessing of datasets.

- `loaders.py`: Contains functions for loading various datasets, including those from the BEIR benchmark.

### `geoIR.eval`

The evaluation module provides tools for assessing the performance of retrieval models.

- `metrics.py`: Implements standard information retrieval metrics.
- `rare.py`: Implements the RARE (Retrieval-Augmented ROUGE) metric.
- `sud.py`: Implements the SUD (Semantic Uniqueness and Diversity) metric.

### `geoIR.geo`

The geo module contains the core geometric components of the library.

- `curvature.py`: Implements functions for computing graph curvature.
- `differentiable.py`: Contains the implementation of the differentiable soft-kNN.
- `graph.py`: Provides functions for building and manipulating k-NN graphs.

### `geoIR.retrieval`

The retrieval module contains the main classes for performing retrieval.

- `retriever.py`: The high-level `GeometricRetriever` class for end-to-end retrieval.
- `encoder.py`: The `Encoder` class for encoding text into embeddings.
- `index.py`: The `Index` class for building and searching the index.

### `geoIR.training`

The training module provides the tools for training and fine-tuning retrieval models.

- `trainer.py`: The `Trainer` class for managing the training process.

## Installation

```bash
# Clone the repository
git clone https://github.com/Intrinsical-AI/geometric-aware-retrieval-v2.git
cd geometric-aware-retrieval-v2

# Install with pip for development
pip install -e ".[dev,hf]"
```

## Quick Start

```python
import geoIR as gi

# 1. Load an encoder
encoder = gi.load_encoder("sentence-transformers/all-MiniLM-L6-v2")

# 2. Define a corpus and build the index
corpus = [
    "The Moon is a natural satellite of Earth.",
    "Mars is often called the Red Planet.",
    "A solar eclipse occurs when the Moon passes between Earth and the Sun.",
]
index = encoder.build_index(corpus, k=2)

# 3. Search the index
query_emb = encoder.encode(["Which planet is red?"])[0]
doc_indices = index.search(query_emb, k=1)

# 4. Geometrical audit
audit = index.geo_audit(curvature=True)
print(audit.curvature)

```

## Current CLI Scope

The CLI is intentionally smaller than the package surface while the public contract is being stabilized.

The supported CLI v0 focus is geometric interpretability and local debugging:

- `geoIR encode`: encode local texts and optionally save `.npy` embeddings.
- `geoIR audit`: inspect geometric properties of an index built from a local corpus.

The deprecated/unsupported transitional command is:

- `geoIR search`

Notes:

- `geoIR search` is intentionally fail-fast in CLI v0 and pending removal in a future clean CLI cut.
- `geoIR audit --plot` is experimental and not part of the supported contract yet.
- For the current architecture and support boundary, see `ARCH.md`.

## Research Benchmark

The BEIR benchmark harness under `research/` is research-only and not part of
the public package contract.

- Exhaustive benchmark spec: `research/BEIR_BENCHMARK_SPEC.md`
- Probe runbook: `research/PROBES.md`
- Result artifacts and aggregation rules: `research/results/beir_euclidean_vs_geo/README.md`

## Makefile Commands

This repository includes a `Makefile` with the following commands:

- `make format`: Format and apply safe lint fixes.
- `make lint`: Check formatting and linting without rewriting files.
- `make type`: Run the type checker.
- `make test`: Run the unit tests.
- `make clean`: Remove temporary files.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{geoIR_2024,
  author = {IntrinsicalAI},
  title = {geoIR: A Python library for geometric-aware information retrieval},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/Intrinsical-AI/geometric-aware-retrieval-v2}}
}
```

## License

Apache-2.0
