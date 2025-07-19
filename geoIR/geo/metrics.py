"""High-level geometric/LLM metrics: RARE, SUD, Contradiction Resilience.

These helpers delegate heavy work to `geoIR.eval.*` but keep dataclass
containers close to numeric utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class MetricResult:  # noqa: D101
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.name}(score={self.score:.3f})"


__all__ = ["MetricResult"]
