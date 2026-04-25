from __future__ import annotations

import pytest
import torch

from geoIR.eval import metrics


def test_evaluate_retrieval_falls_back_without_pytrec_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "pytrec_eval", None)

    ndcg, recall = metrics.evaluate_retrieval(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        {"q1": {"d1": 1}},
        ["d1", "d2"],
        k_eval=1,
    )

    assert ndcg == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_ppr_reranking_clamps_topk_to_available_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "pytrec_eval", None)

    ndcg, recall = metrics.evaluate_retrieval_ppr(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        {"q1": {"d1": 1}},
        ["d1", "d2", "d3"],
        torch.tensor(
            [
                [0.0, 0.8, 0.2],
                [0.8, 0.0, 0.2],
                [0.2, 0.8, 0.0],
            ],
            dtype=torch.float32,
        ),
        k_eval=1,
        topk=100,
    )

    assert 0.0 <= ndcg <= 1.0
    assert 0.0 <= recall <= 1.0
