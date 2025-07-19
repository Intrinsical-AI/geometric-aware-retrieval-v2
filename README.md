# Geometric-Aware Retrieval with Differentiable Soft-kNN

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


## Usage Examples

### 1. Fine-tuning a Model

You can fine-tune your own models using the trainer

### 2. IR metrics - RARE, SUD, non_monotonicity
