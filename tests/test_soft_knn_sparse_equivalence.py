"""The row-blocked sparse soft-kNN must reproduce the dense graph and its downstream metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from geoIR.eval.metrics import (
    _sub_adjacency,
    evaluate_retrieval_soft_local,
    graph_distribution_metrics,
)
from geoIR.geo.differentiable import soft_knn_graph, soft_knn_graph_sparse

MODULE_PATH = Path(__file__).resolve().parents[1] / "research" / "beir_euclidean_vs_geo.py"
pytestmark = pytest.mark.filterwarnings("ignore:Embeddings do not require gradients")


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("beir_sparse_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clustered(n: int, dim: int = 16, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(8, dim, generator=generator) * 3
    labels = torch.randint(0, 8, (n,), generator=generator)
    points = centers[labels] + torch.randn(n, dim, generator=generator)
    return torch.nn.functional.normalize(points, dim=1)


def _both(embeddings: torch.Tensor, k: int, block_rows: int = 64):
    torch.manual_seed(0)
    W_dense, A_dense, diag_dense = soft_knn_graph(
        embeddings, k=k, return_adjacency=True, return_diagnostics=True
    )
    torch.manual_seed(0)
    W_sparse, A_sparse, diag_sparse = soft_knn_graph_sparse(embeddings, k=k, block_rows=block_rows)
    return (W_dense, A_dense, diag_dense), (W_sparse, A_sparse, diag_sparse)


@pytest.mark.parametrize(("n", "k"), [(150, 5), (400, 10), (2100, 20)])
def test_sparse_graph_matches_dense(n: int, k: int) -> None:
    (W_d, A_d, diag_d), (W_s, A_s, diag_s) = _both(_clustered(n), k)
    assert sp.issparse(A_s) and sp.issparse(W_s)
    dense_pattern = (A_d != 0).numpy()
    assert np.array_equal(dense_pattern, A_s.toarray() != 0)
    np.testing.assert_allclose(A_s.toarray(), A_d.numpy(), rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(W_s.toarray(), W_d.numpy(), rtol=2e-5, atol=1e-6)
    for key in ("gamma_used", "entropy", "effective_degree", "actual_degree"):
        assert diag_s[key] == pytest.approx(diag_d[key], rel=1e-5)
    assert diag_s["status"] == diag_d["status"]


def test_downstream_metrics_match_dense() -> None:
    module = _benchmark_module()
    k = 10
    embeddings = _clustered(400, seed=1)
    (_, A_d, _), (_, A_s, _) = _both(embeddings, k)
    # Consumers (neighbor purity, edge overlap) use each row's neighbors as a set.
    sparse_rows = module.adjacency_topk_neighbors(A_s, k)
    dense_rows = module.adjacency_topk_neighbors(A_d, k)
    assert [set(row) for row in sparse_rows] == [set(row) for row in dense_rows]
    # Rows are renormalized to degree k, so the std is float32 noise around zero.
    assert module.safe_degree_std(A_s) == pytest.approx(module.safe_degree_std(A_d), abs=1e-5)
    dense_metrics = graph_distribution_metrics(A_d)
    sparse_metrics = graph_distribution_metrics(A_s)
    for key, value in dense_metrics.items():
        assert sparse_metrics[key] == pytest.approx(value, rel=1e-5)


def test_eval_on_sparse_matrix_equals_eval_on_its_dense_copy() -> None:
    embeddings = _clustered(300, seed=2)
    torch.manual_seed(0)
    _, A_s, _ = soft_knn_graph_sparse(embeddings, k=8, block_rows=50)
    A_dense_copy = torch.from_numpy(A_s.toarray())
    idx = torch.tensor([5, 17, 3, 250, 120])
    assert torch.equal(_sub_adjacency(A_s, idx), _sub_adjacency(A_dense_copy, idx))
    queries = _clustered(12, seed=3)
    doc_ids = [f"d{i}" for i in range(300)]
    qrels = {str(q): {doc_ids[(7 * q) % 300]: 1} for q in range(12)}
    sparse_result = evaluate_retrieval_soft_local(queries, embeddings, qrels, doc_ids, A_s, topk=20)
    dense_result = evaluate_retrieval_soft_local(
        queries, embeddings, qrels, doc_ids, A_dense_copy, topk=20
    )
    assert sparse_result == dense_result


def test_blocked_dense_topk_neighbors_match_full_matrix() -> None:
    module = _benchmark_module()
    embeddings = _clustered(333, seed=4)
    full = module.dense_topk_neighbors(embeddings, 12)
    blocked = module.dense_topk_neighbors(embeddings, 12, block_rows=50)
    assert [set(row) for row in blocked] == [set(row) for row in full]


def test_cli_defaults_to_dense_and_accepts_sparse() -> None:
    parser = _benchmark_module().build_argument_parser()
    assert parser.parse_args([]).soft_graph_impl == "dense"
    assert parser.parse_args(["--soft-graph-impl", "sparse"]).soft_graph_impl == "sparse"


@pytest.mark.parametrize("kind", ["zero", "tied", "few_neighbors", "rounding_ties"])
def test_sparse_topk_preserves_dense_ties_and_implicit_zeros(kind: str) -> None:
    module = _benchmark_module()
    adjacency = torch.zeros(12, 12)
    if kind == "tied":
        adjacency.fill_(1)
    elif kind == "few_neighbors":
        adjacency[torch.arange(12), (torch.arange(12) + 1) % 12] = 1
    elif kind == "rounding_ties":
        adjacency = torch.ones(12, 12, dtype=torch.float64) + torch.arange(12) * 1e-10
    expected = module.adjacency_topk_neighbors(adjacency, 4)
    assert module.adjacency_topk_neighbors(sp.csr_matrix(adjacency.numpy()), 4) == expected


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_sparse_rejects_nonfinite_embeddings(bad_value: float) -> None:
    embeddings = torch.randn(6, 3)
    embeddings[0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        soft_knn_graph_sparse(embeddings, k=2, gamma=0.2)


@pytest.mark.parametrize("shape", [(0, 3), (1, 3)])
def test_sparse_rejects_insufficient_points(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="at least two"):
        soft_knn_graph_sparse(torch.zeros(shape), k=2, gamma=0.2)


def test_sparse_rejects_degenerate_automatic_calibration() -> None:
    with pytest.raises(ValueError, match="positive distances"):
        soft_knn_graph_sparse(torch.zeros(6, 3), k=2)


@pytest.mark.parametrize("gamma", [0.0, -0.2, float("nan"), float("inf")])
def test_sparse_rejects_invalid_gamma(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma must be finite and positive"):
        soft_knn_graph_sparse(_clustered(6), k=2, gamma=gamma)


@pytest.mark.parametrize(("k", "block_rows"), [(0, 3), (-1, 3), (2, 0), (2, -1)])
def test_sparse_rejects_nonpositive_sizes(k: int, block_rows: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        soft_knn_graph_sparse(_clustered(6), k=k, gamma=0.2, block_rows=block_rows)


def test_sparse_requires_explicit_gamma_for_k_one() -> None:
    with pytest.raises(ValueError, match="requires k > 1"):
        soft_knn_graph_sparse(_clustered(6), k=1)


def test_sparse_rejects_overflowed_distances() -> None:
    embeddings = torch.tensor([[1e30, 0], [-1e30, 0]])
    with pytest.raises(ValueError, match="distances must be finite"):
        soft_knn_graph_sparse(embeddings, k=1, gamma=0.2)


def test_explicit_gamma_preserves_all_ties_and_disables_gradients() -> None:
    embeddings = torch.zeros(12, 3, requires_grad=True)
    W_d, A_d = soft_knn_graph(embeddings, k=2, gamma=0.2, return_adjacency=True)
    W_s, A_s = soft_knn_graph_sparse(
        embeddings, k=2, gamma=0.2, block_rows=3, return_diagnostics=False
    )
    assert A_s.nnz == 12 * 11  # Keeping all tied edges can exceed N*k.
    np.testing.assert_allclose(A_s.toarray(), A_d.detach().numpy(), rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(W_s.toarray(), W_d.detach().numpy(), rtol=2e-5, atol=1e-7)
    assert embeddings.grad is None


def test_explicit_gamma_bounds_distance_blocks(monkeypatch) -> None:
    shapes: list[tuple[int, int]] = []
    original = torch.cdist

    def cdist(left, right, *args, **kwargs):
        shapes.append((left.shape[0], right.shape[0]))
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", cdist)
    soft_knn_graph_sparse(_clustered(128), k=4, gamma=0.2, block_rows=17)
    assert shapes and all(rows <= 17 and cols == 128 for rows, cols in shapes)
