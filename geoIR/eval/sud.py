"""Semantic Utility Delta (SUD).

SUD measures the marginal utility of a set of *new* documents compared to a
set of *ground-truth* documents, as judged by an LLM ensemble.

    SUD = score(LLM(query, new_docs)) - score(LLM(query, gt_docs))

This metric helps quantify if a retriever finds useful (but unlabelled) information.
"""

from __future__ import annotations

from typing import List

from geoIR.eval.judges import BaseJudge, judge_ensemble

from .metrics import MetricResult


def SUD(
    query: str,
    gt_docs: List[str],
    new_docs: List[str],
    *,
    judges: List[BaseJudge] | None = None,
    policy: str = "mean",
) -> MetricResult:  # noqa: D401
    """Calculate the Semantic Utility Delta using an LLM judge ensemble."""
    # Score the answer generated from the new documents
    res_new = judge_ensemble(query, new_docs, judges=judges, policy=policy)

    # Score the answer generated from the ground-truth documents
    res_gt = judge_ensemble(query, gt_docs, judges=judges, policy=policy)

    # The delta is the difference in scores
    delta = res_new.score - res_gt.score

    details = {
        "score_new": res_new.score,
        "score_gt": res_gt.score,
        "details_new": res_new.details,
        "details_gt": res_gt.details,
    }
    return MetricResult(name="SUD", score=float(delta), details=details)
