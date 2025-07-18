# Geometric-Aware Retrieval (geoIR)

This repository provides a Python framework for fine-tuning and evaluating sentence embedding models using geometric regularization techniques. The core idea is to represent the embedding space as a manifold and use its curvature properties to improve retrieval performance, especially for queries requiring nuanced understanding.

The main contributions are:
- **InfoNCE-Geo Loss**: A contrastive loss function that operates on geodesic distances over a k-NN graph of document embeddings.
- **Geometric Regularization**: Using Ricci and Forman curvature as regularizers to encourage a more structured embedding space.
- **High-Level API**: A simple `GeometricRetriever` class for easy indexing and searching.

## Installation

To install the necessary dependencies, clone the repository and install it in editable mode:

```bash
git clone https://github.com/your-username/geometric-aware-retrieval-v2.git
cd geometric-aware-retrieval-v2
pip install -e .
```

## Quick Start

Using the library is straightforward with the `GeometricRetriever` API. Here's a simple example:

```python
from geoIR.retrieval.retriever import GeometricRetriever

# 1. Define a small corpus of documents
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "A man is riding a white horse on an enclosed track.",
    "A monkey is playing drums.",
]

# 2. Initialize the retriever
# This will download a pre-trained model from HuggingFace
retriever = GeometricRetriever("sentence-transformers/all-MiniLM-L6-v2")

# 3. Index the corpus (builds the k-NN graph and computes geodesic distances)
print("Building index...")
retriever.index(corpus, k_graph=3, verbose=True)

# 4. Define a query and search
query = "A man is riding an animal."
print(f"\nQuery: '{query}'")

# Search using geodesic distance on the graph
print("\n--- Searching with Geodesic Distance ---")
geodesic_results = retriever.search(query, top_k=3, metric="geodesic")
for i, doc_idx in enumerate(geodesic_results):
    print(f"{i+1}: {corpus[doc_idx]} (doc_id={doc_idx})")

# Search using standard cosine similarity
print("\n--- Searching with Cosine Similarity ---")
cosine_results = retriever.search(query, top_k=3, metric="cosine")
for i, doc_idx in enumerate(cosine_results):
    print(f"{i+1}: {corpus[doc_idx]} (doc_id={doc_idx})")
```

## Fine-tuning a Model

You can fine-tune your own models using the unified training script. Configuration is handled via YAML files.

1.  **Create a configuration file** (e.g., `my_experiment.yaml`):

    ```yaml
    # my_experiment.yaml
    dataset: "/path/to/your/triplets.tsv" # Or path to a corpus file
    trainer: "geo" # or "classic"

    encoder:
      model_name: "sentence-transformers/all-MiniLM-L6-v2"

    geo_params:
      epochs: 5
      batch_size: 8
      lr: 1e-5
      k_graph: 5
      lambda_ricci: 0.5

    output_dir: "runs/my_first_experiment"
    ```

2.  **Run the fine-tuning script**:

    ```bash
    python -m geoIR.scripts.finetune --config my_experiment.yaml
    ```

The trained model checkpoint and logs will be saved in the specified `output_dir`.
