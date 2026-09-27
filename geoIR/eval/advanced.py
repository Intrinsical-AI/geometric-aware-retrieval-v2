"""Additional experimental LLM-centric retrieval metrics.

This module hosts *prototype* implementations of the metrics described in the
project notes. They are intentionally lightweight so that they work without
external dependencies or API keys. Replace the placeholder logic with proper
LLM calls / statistical analysis as needed.

Metrics implemented
-------------------
1. Non-Monotonicity Score (NMS)
2. Meta-Evaluation (META)
3. Contradiction Resilience (CoRe)

All metrics return the common ``MetricResult`` dataclass so they can be mixed
and aggregated with RARE and SUD.
"""

from __future__ import annotations

from itertools import combinations
from statistics import mean
from typing import List, Sequence

from geoIR.eval.metrics import MetricResult

__all__ = [
    "non_monotonicity_score",
    "meta_evaluation",
    "contradiction_resilience",
]


# ---------------------------------------------------------------------------
# 1. Non-Monotonicity Score
# ---------------------------------------------------------------------------


def non_monotonicity_score(recalls: Sequence[float], qualities: Sequence[float]) -> MetricResult:  # noqa: D401
    """Compute how often *lower* recall leads to *higher* answer quality.

    Given two aligned sequences::
        recalls   = [r_1,  r_2,  ..., r_n]
        qualities = [q_1,  q_2,  ..., q_n]

    the metric counts the number of pairs ``(i, j)`` with ``r_i < r_j`` but
    ``q_i > q_j`` – i.e. non-monotonic improvements – divided by the total
    number of comparable pairs ``n*(n-1)/2``.
    A higher score (→ **1.0**) indicates *more* violations of the expected
    "higher recall → better answer" assumption.
    """
    assert len(recalls) == len(qualities), "recall/quality length mismatch"
    n_pairs = 0
    violations = 0
    for i, j in combinations(range(len(recalls)), 2):
        n_pairs += 1
        if recalls[i] < recalls[j] and qualities[i] > qualities[j]:
            violations += 1
    score = violations / n_pairs if n_pairs else 0.0
    return MetricResult(
        name="NMS", score=float(score), details={"violations": violations, "pairs": n_pairs}
    )


# ---------------------------------------------------------------------------
# 2. Meta-Evaluation
# ---------------------------------------------------------------------------


def meta_evaluation(judgments: Sequence[MetricResult]) -> MetricResult:  # noqa: D401
    """Aggregate a list of *judge* results (e.g. from LLM-as-a-judge).

    Currently returns the mean of the individual scores. The *details* field
    stores per-metric scores for further analysis.
    """
    if not judgments:
        return MetricResult(name="META", score=0.0)
    scores = [j.score for j in judgments]
    meta_score = mean(scores)
    details = {j.name: j.score for j in judgments}
    return MetricResult(name="META", score=float(meta_score), details=details)


# ---------------------------------------------------------------------------
# 3. Contradiction Resilience
# ---------------------------------------------------------------------------

_NEGATION_MARKERS = {"not", "no", "never", "nor", "none", "nobody", "nothing"}


def _count_negations(text: str) -> int:
    return sum(word.lower() in _NEGATION_MARKERS for word in text.split())


def contradiction_resilience(docs: List[str]) -> MetricResult:  # noqa: D401
    """Toy proxy for contradiction handling.

    We assume documents with *many* negations are more likely to contain
    conflicting statements. The retriever / generator is considered resilient
    if it presents a *consistent* answer despite such inputs.

    The placeholder implementation simply computes 1 / (1 + negations).
    A real setup would call an LLM to check if contradictions are resolved.
    """
    n_negations = sum(_count_negations(d) for d in docs)
    score = 1.0 / (1.0 + n_negations)
    return MetricResult(name="CoRe", score=float(score), details={"negations": n_negations})
