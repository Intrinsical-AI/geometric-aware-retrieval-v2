"""Unit tests for the high-level GeometricRetriever API."""

import pytest
from geoIR.retrieval.retriever import GeometricRetriever


@pytest.fixture(scope="module")
def small_corpus():
    """Provides a small, reusable corpus for testing."""
    return [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "A man is riding a white horse on an enclosed track.",
    ]


@pytest.fixture(scope="module")
def initialized_retriever():
    """Provides a retriever initialized with a standard model."""
    return GeometricRetriever("sentence-transformers/all-MiniLM-L6-v2")


def test_retriever_initialization(initialized_retriever):
    """Test that the retriever and its encoder are initialized correctly."""
    assert initialized_retriever is not None
    assert initialized_retriever.encoder is not None
    assert initialized_retriever.corpus_embeddings is None


def test_indexing(initialized_retriever, small_corpus):
    """Test the indexing process."""
    retriever = initialized_retriever
    retriever.index(small_corpus, k_graph=2, verbose=False)
    
    assert retriever.corpus_embeddings is not None
    assert retriever.corpus_embeddings.shape[0] == len(small_corpus)
    assert retriever.corpus_graph is not None
    assert retriever.distance_matrix is not None


def test_search_raises_error_if_not_indexed():
    """Test that searching before indexing raises a RuntimeError."""
    retriever = GeometricRetriever("sentence-transformers/all-MiniLM-L6-v2")
    with pytest.raises(RuntimeError):
        retriever.search("a query", metric="geodesic")


def test_search_metrics(initialized_retriever, small_corpus):
    """Test that both geodesic and cosine search return valid results."""
    retriever = initialized_retriever
    # Ensure index is built before searching
    if retriever.corpus_embeddings is None:
        retriever.index(small_corpus, k_graph=2)

    query = "A man on a horse"
    top_k = 3

    # Test geodesic search
    geodesic_results = retriever.search(query, top_k=top_k, metric="geodesic")
    assert isinstance(geodesic_results, list)
    assert len(geodesic_results) == top_k
    assert all(isinstance(i, int) for i in geodesic_results)

    # Test cosine search
    cosine_results = retriever.search(query, top_k=top_k, metric="cosine")
    assert isinstance(cosine_results, list)
    assert len(cosine_results) == top_k
    assert all(isinstance(i, int) for i in cosine_results)


    # Test invalid metric
    with pytest.raises(ValueError):
        retriever.search(query, top_k=top_k, metric="invalid_metric")
