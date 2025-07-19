# Geometric-Aware Retrieval with Differentiable Soft-kNN

A Python library for geometric-aware information retrieval using differentiable k-nearest neighbors and graph-based re-ranking.

## Features

- **Geometric Retrieval**: Leverage graph-based distances for more semantically meaningful similarity search
- **Differentiable Pipeline**: End-to-end trainable retrieval models
- **Multiple Backends**: Supports FAISS for fast similarity search
- **Pre-trained Models**: Easy integration with HuggingFace's sentence-transformers
- **Evaluation**: Built-in support for standard IR metrics and geometric metrics (RARE, SUD)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/geometric-aware-retrieval-v2.git
cd geometric-aware-retrieval-v2

# Install with pip
pip install -e .
```

## Quick Start

### Basic Usage

```python
from geoIR.retrieval.retriever import GeometricRetriever

# Initialize with a pre-trained model
retriever = GeometricRetriever("sentence-transformers/all-MiniLM-L6-v2")

# Index documents
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "A man is riding a white horse on an enclosed track.",
]

retriever.index(corpus, k_graph=3)

# Search with geodesic distance
results = retriever.search("A man is riding an animal.", top_k=3, metric="geodesic")
```

### Advanced Usage

#### Custom Indexing

```python
from geoIR.retrieval.encoder import load_encoder
from geoIR.retrieval.index import Index

# Load a custom encoder
encoder = load_encoder("sentence-transformers/all-mpnet-base-v2", device="cuda")

# Encode documents
embeddings = encoder.encode(corpus)

# Build custom index
index = Index(k=10, metric="cosine")
index.build(embeddings)

# Save and load
index.save("my_index.pkl")
loaded_index = Index.load("my_index.pkl")
```

#### Fine-tuning

```python
from geoIR.training.trainer import Trainer
from geoIR.data.loaders import load_beir_dataset

# Load dataset
train_loader, val_loader = load_beir_dataset("msmarco")

# Initialize trainer
trainer = Trainer(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir="runs/experiment_1"
)

# Start training
trainer.train(epochs=10, learning_rate=2e-5)
```

## API Reference

### Core Classes

#### `GeometricRetriever`

Main class for end-to-end retrieval.

```python
retriever = GeometricRetriever(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None
)
```

**Methods**:
- `index(documents: List[str], k_graph: int = 10, **kwargs) -> None`: Index documents
- `search(query: str, top_k: int = 10, metric: str = "geodesic") -> List[int]`: Search documents
- `save(path: str) -> None`: Save retriever state
- `load(path: str) -> 'GeometricRetriever'`: Load retriever state

#### `Index`

Low-level index for similarity search.

```python
index = Index(
    k: int = 10,
    metric: str = "cosine",
    **kwargs
)
```

**Methods**:
- `build(embeddings: np.ndarray) -> None`: Build index from embeddings
- `search(query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]`: Search index
- `save(path: str) -> None`: Save index
- `load(path: str) -> 'Index'`: Load index

## Evaluation

### Standard IR Metrics

```python
from geoIR.eval.metrics import evaluate_retrieval

results = evaluate_retrieval(
    query_embeddings,  # [n_queries, dim]
    doc_embeddings,    # [n_docs, dim]
    qrels,             # Dict[query_id, Dict[doc_id, relevance]]
    metrics=["map", "ndcg@10", "recall@100"]
)
```

### Geometric Metrics

```python
from geoIR.eval.rare import compute_rare
from geoIR.eval.sud import compute_sud

rare_score = compute_rare(index, k=10)
sud_score = compute_sud(index, k=10)
```

## Examples

See the `examples/` directory for more detailed usage:

- `basic_usage.py`: Basic retrieval pipeline
- `quickstart.py`: Quick start guide
- `quick_experiment_cli.py`: Command-line interface for experiments
- `differentiable_demo.py`: Differentiable retrieval demo

## Citation

If you use this library in your research, please cite:

```bibtex
@software{geometric_ir_2023,
  author = {Your Name},
  title = {Geometric-Aware Retrieval},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/geometric-aware-retrieval-v2}}
}
```

## License

MIT
