"""The candidate sub-adjacency gather must equal the original two-step indexing exactly."""

from __future__ import annotations

import pytest
import torch

from geoIR.eval.metrics import _sub_adjacency


@pytest.mark.parametrize("seed", range(20))
def test_sub_adjacency_matches_row_then_column_indexing(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    n = int(torch.randint(1, 400, (1,), generator=generator))
    A = torch.rand((n, n), generator=generator)
    A[A < 0.7] = 0.0
    sims = torch.rand(n, generator=generator)
    for topk in (1, min(20, n), min(100, n), n):
        _, idx = sims.topk(topk)
        assert torch.equal(_sub_adjacency(A, idx), A[idx][:, idx])
