"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Scoring

File        : scorer.py

Description
-----------
Computes the final Belief Stability Score from belief
persistence profiles.

Version 1 Formula
-----------------
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


class BeliefScorer:
    """
    Computes the final Belief Stability Score.
    """

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
            )

        scores = [
            self._belief_score(profile)
            for profile in profiles
        ]

        stability_score = sum(scores) / len(scores)

        return BeliefStabilityResult(
            stability_score=stability_score,
            profiles=profiles,
        )