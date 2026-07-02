"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Scoring

File        : base.py

Description
-----------
Shared interface for every scoring strategy, so pipeline.py
can swap scorers by name (``build_scorer(method)``) without
changing anything else in the pipeline.
---------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from belief_stability.models import (
    BeliefProfile,
    BeliefStabilityResult,
)


class BaseBeliefScorer(ABC):

    method_name: str = "base"

    @abstractmethod
    def compute(
        self,
        profiles: List[BeliefProfile],
    ) -> BeliefStabilityResult:
        raise NotImplementedError


def aggregate(
    result: BeliefStabilityResult,
    method: str = "mean",
) -> float:
    """
    Re-aggregate a BeliefStabilityResult's per-belief scores
    into one scalar, using ``method`` instead of the
    scorer's default (mean).

    "min" is the pessimistic aggregation used at sentence
    level: the single worst (most contradicted) belief in a
    sentence drags the whole sentence's score down, rather
    than being averaged out by other beliefs in the same
    sentence.
    """

    if not result.profiles:
        return result.stability_score

    scores = [
        p.score if p.score is not None else 0.0
        for p in result.profiles
    ]

    if method == "min":
        return min(scores)

    if method == "mean":
        return sum(scores) / len(scores)

    raise ValueError(f"Unknown aggregation method '{method}'.")
