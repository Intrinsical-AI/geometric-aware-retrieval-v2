"""LLM-as-a-judge helpers and simple ensembling policies.

This module defines a minimal interface so we can plug different judge engines
(OpenAI, local Llama-cpp, HF pipelines, etc.) and combine their opinions via
*policies* (mean, majority, max, etc.).  Actual API calls are optional – if the
required packages or keys are missing, the judge will raise a clear error so
callers can skip/replace it.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from statistics import mean
from typing import List

from .metrics import MetricResult

__all__ = [
    "BaseJudge",
    "OpenAIJudge",
    "HFJudge",
    "MockJudge",
    "aggregate_scores",
    "judge_ensemble",
    "make_judges",
]


class BaseJudge(ABC):  # noqa: D101
    name: str = "LLM_JUDGE"

    @abstractmethod
    def __call__(self, question: str, docs: List[str], answer: str | None = None) -> MetricResult:  # noqa: D401
        """Return a MetricResult with *higher* meaning *better*."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class OpenAIJudge(BaseJudge):  # noqa: D101
    name = "OpenAI"

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str | None = None):
        try:
            import openai  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("openai package not installed – install or use MockJudge") from exc
        self._openai = openai
        self._model = model
        self._openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._openai.api_key:  # pragma: no cover
            raise RuntimeError("OpenAI API key missing; set OPENAI_API_KEY env var")

    def __call__(self, question: str, docs: List[str], answer: str | None = None) -> MetricResult:  # noqa: D401
        answer = answer or " ".join(docs)[:1000]
        prompt = (
            "You are an expert grader. Given the question, reference docs and an answer, "
            "return a score between 0 and 1 reflecting answer quality."
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"},
        ]
        resp = self._openai.ChatCompletion.create(
            model=self._model, messages=messages, temperature=0.0
        )
        # Expect the model to respond with a float in [0,1]
        try:
            score = float(resp.choices[0].message["content"].strip())
        except Exception:  # pragma: no cover
            score = 0.0
        return MetricResult(name=self.name, score=score, details={"raw": resp})


class HFJudge(BaseJudge):  # noqa: D101
    """Judge using a local HuggingFace *text-generation* model.

    The model is expected to return a float score when prompted. We keep the
    interface identical to OpenAIJudge so users can swap backends.
    """

    name = "HF"

    def __init__(self, model: str = "gpt2", device: int | str | None = None):
        try:
            from transformers import pipeline  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers package not installed – install or use MockJudge"
            ) from exc
        kwargs = {"model": model, "device": device} if device is not None else {"model": model}
        self._pipe = pipeline("text-generation", **kwargs)

    def __call__(self, question: str, docs: List[str], answer: str | None = None) -> MetricResult:  # noqa: D401
        answer = answer or " ".join(docs)[:1000]
        prompt = (
            "You are an expert grader. Given the question and answer, respond with a float between 0 and 1 "
            "reflecting answer quality.\n\nQuestion:" + question + "\nAnswer:" + answer + "\nScore:"  # noqa: E501
        )
        try:
            gen = self._pipe(prompt, max_new_tokens=4, do_sample=False)[0]["generated_text"]
            # parse last float in output
            score_str = gen.split()[-1].strip(" ,;\n")
            score = float(score_str)
        except Exception:  # pragma: no cover
            score = 0.0
        return MetricResult(name=self.name, score=score, details={})


class MockJudge(BaseJudge):  # noqa: D101
    """Fallback judge: keyword overlap same as previous prototype."""

    name = "MOCK"
    _KEYWORDS = {"fact", "reason", "evidence", "citation"}

    def __call__(self, question: str, docs: List[str], answer: str | None = None) -> MetricResult:  # noqa: D401
        answer = answer or " ".join(docs)[:1000]
        hits = sum(kw in answer.lower() for kw in self._KEYWORDS)
        score = hits / len(self._KEYWORDS)
        return MetricResult(name=self.name, score=float(score), details={"keywords_hit": hits})


# ---------------------------------------------------------------------------
# Aggregation / Voting
# ---------------------------------------------------------------------------


def aggregate_scores(results: List[MetricResult], policy: str = "mean") -> float:  # noqa: D401
    """Combine multiple judge scores into a single value.

    Supported *policy* values:
    • "mean" (default) – arithmetic mean.
    • "majority" – vote > 0.5 wins, proportion of positive votes.
    • "max" – optimistic, take best score.
    • "min" – pessimistic.
    """
    if not results:
        return 0.0
    scores = [r.score for r in results]
    policy = policy.lower()
    if policy == "mean":
        return mean(scores)
    if policy == "majority":
        positives = sum(s > 0.5 for s in scores)
        return positives / len(scores)
    if policy == "max":
        return max(scores)
    if policy == "min":
        return min(scores)
    raise ValueError(f"Unknown aggregation policy: {policy}")


def judge_ensemble(
    question: str,
    docs: List[str],
    *,
    answer: str | None = None,
    judges: List[BaseJudge] | None = None,
    policy: str = "mean",
) -> MetricResult:  # noqa: D401
    """Run *judges* and aggregate their scores into a single MetricResult."""
    judges = judges or [MockJudge()]
    results = [j(question, docs, answer) for j in judges]
    score = aggregate_scores(results, policy)
    details = {r.name: r.score for r in results}
    details["policy"] = policy
    return MetricResult(name="LLM_ENSEMBLE", score=float(score), details=details)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_judges(names: str | List[str]) -> List[BaseJudge]:  # noqa: D401
    """Instantiate judges from *names* string or list.

    Examples::
        make_judges("openai,mock")
        make_judges(["hf:gpt2", "mock"])
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    result: List[BaseJudge] = []
    for spec in names:
        if spec.lower().startswith("openai"):
            # allow "openai" or "openai:model"
            parts = spec.split(":", 1)
            model = parts[1] if len(parts) == 2 else "gpt-3.5-turbo"
            result.append(OpenAIJudge(model=model))
        elif spec.lower().startswith("hf"):
            parts = spec.split(":", 1)
            model = parts[1] if len(parts) == 2 else "gpt2"
            result.append(HFJudge(model=model))
        elif spec.lower() in {"mock", "keyword", "dummy"}:
            result.append(MockJudge())
        else:  # pragma: no cover
            raise ValueError(f"Unknown judge spec: {spec}")
    return result
