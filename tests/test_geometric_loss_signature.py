"""geometric_loss_end_to_end must accept the trainer's call and forward the soft-kNN gamma."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import torch

from geoIR.geo import differentiable
from geoIR.geo.differentiable import geometric_loss_end_to_end

TRAINER = Path(__file__).resolve().parents[1] / "geoIR" / "training" / "trainer.py"


def _trainer_call() -> ast.Call:
    calls = [
        node
        for node in ast.walk(ast.parse(TRAINER.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "geometric_loss_end_to_end"
    ]
    assert len(calls) == 1
    return calls[0]


def test_trainer_call_binds_to_signature() -> None:
    call = _trainer_call()
    keywords = {keyword.arg: None for keyword in call.keywords}
    assert "gamma" in keywords
    inspect.signature(geometric_loss_end_to_end).bind(*([None] * len(call.args)), **keywords)


def _embeddings(batch: int = 2, negatives: int = 1, dim: int = 4):
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(batch, dim, generator=generator, requires_grad=True)
    p = torch.randn(batch, dim, generator=generator, requires_grad=True)
    n = torch.randn(batch, negatives, dim, generator=generator, requires_grad=True)
    return q, p, n


def test_gamma_is_forwarded_to_soft_knn_graph(monkeypatch) -> None:
    seen: list[float | None] = []
    original = differentiable.soft_knn_graph

    def spy(embeddings, k=8, gamma=None, **kwargs):
        seen.append(gamma)
        return original(embeddings, k=k, gamma=gamma, **kwargs)

    monkeypatch.setattr(differentiable, "soft_knn_graph", spy)
    geometric_loss_end_to_end(*_embeddings(), k_graph=3, gamma=0.2)
    geometric_loss_end_to_end(*_embeddings(), k_graph=3)
    assert seen == [0.2, None]


def test_scheduled_gamma_gives_finite_loss() -> None:
    loss, metrics = geometric_loss_end_to_end(*_embeddings(), k_graph=3, gamma=0.1)
    assert math.isfinite(loss.item())
    assert math.isfinite(metrics["loss_total"])
