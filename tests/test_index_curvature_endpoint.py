"""Public alpha endpoints must remain finite, including zero-curvature nodes."""

import networkx as nx
import numpy as np

from geoIR.retrieval.index import Index


def test_pure_curvature_ranks_zero_between_positive_and_negative() -> None:
    index = Index(np.eye(3), ["negative", "zero", "positive"], nx.path_graph(3))
    index._avg_curv = {0: -1.0, 1: 0.0, 2: 1.0}
    assert index.search(np.array([1.0, 0.0, 0.0]), k=3, alpha=1.0) == [2, 1, 0]


def test_zero_alpha_preserves_cosine_ranking() -> None:
    index = Index(np.eye(3), ["negative", "zero", "positive"], nx.path_graph(3))
    index._avg_curv = {0: -1.0, 1: 0.0, 2: 1.0}
    query = np.array([0.9, 0.3, 0.1])
    assert index.search(query, k=3, alpha=0.0) == index.search(query, k=3) == [0, 1, 2]
