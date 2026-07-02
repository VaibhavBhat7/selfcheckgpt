"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Scoring

File        : baseline_scorer.py

Description
-----------
Version 1 formula, kept unchanged as the control arm for
the baseline vs. Bayesian vs. graph ablation.

BeliefScore =

(Support - Contradict) /
(Support + Absent + Contradict)

Overall Stability Score

=

Average(BeliefScore)

---------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from belief_stability.models import (
    BeliefProfile,
    BeliefStabilityResult,
)
from .base import BaseBeliefScorer


class BaselineScorer(BaseBeliefScorer):
    """
    Computes the Version-1 Belief Stability Score.
    """

    method_name = "baseline"

    @staticmethod
    def _belief_score(profile: BeliefProfile) -> float:

        total = (
            profile.support +
            profile.absent +
            profile.contradict
        )

        if total == 0:
            return 0.0

        return (
            profile.support -
            profile.contradict
        ) / total

    def compute(
        self,
        profiles: List[BeliefProfile],
    ) -> BeliefStabilityResult:
        """
        Compute the overall belief stability score.

        Parameters
        ----------
        profiles : List[BeliefProfile]

        Returns
        -------
        BeliefStabilityResult
        """

        if not profiles:

            return BeliefStabilityResult(
                stability_score=0.0,
                profiles=[],
                method=self.method_name,
            )

        scores = []

        for profile in profiles:

            belief_score = self._belief_score(profile)

            profile.score = belief_score

            scores.append(belief_score)

        stability_score = sum(scores) / len(scores)

        return BeliefStabilityResult(
            stability_score=stability_score,
            profiles=profiles,
            method=self.method_name,
        )


# Backward-compatible alias (Version 1 name).
BeliefScorer = BaselineScorer
