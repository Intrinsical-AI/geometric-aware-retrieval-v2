"""Response-Aware Retrieval Effectiveness (RARE).

Prototype: uses simple ROUGE-L overlap between generated answer and reference.
LLM-as-a-judge integration to come.
"""
from __future__ import annotations

from typing import List

from geoIR.eval.judges import judge_ensemble

try:
    from rouge_score import rouge_scorer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    rouge_scorer = None

from geoIR.geo.metrics import MetricResult


def RARE(query: str, docs: List[str], reference: str | None = None) -> MetricResult:  # noqa: D401
    """Dummy RARE metric: prompt concat → answer via naive join, score ROUGE-L."""
    answer = " ".join(docs)[:1000]  # truncate
    if reference is None:
        # Delegate to LLM ensemble if no reference answer available
        res = judge_ensemble(query, docs, answer=answer)
        res.name = "RARE"
        return res
    # --- reference-based scoring branch ---
    if rouge_scorer is None:
        score = float(len(set(answer.split()) & set(reference.split())) / max(1, len(reference.split())))
    else:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(reference, answer)["rougeL"].fmeasure
    return MetricResult(name="RARE", score=float(score), details={"answer": answer})
