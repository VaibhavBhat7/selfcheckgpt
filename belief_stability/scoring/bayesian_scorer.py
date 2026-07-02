"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Scoring

File        : bayesian_scorer.py

Description
-----------
Models each belief's persistence as a Beta-Binomial
posterior instead of a flat linear combination.

alpha = 1 + support_weight
beta  = 1 + contradict_weight + absent_discount * absent_weight

posterior_mean = alpha / (alpha + beta)

``support_weight``/``contradict_weight`` are match-tier
confidence-weighted counts (see persistence.py), so a
low-confidence semantic match contributes less evidence
than an exact match. ABSENT is treated as partial,
discounted evidence of instability rather than ignored
(discount=0) or fully penalized like a contradiction
(discount=1) - ``absent_discount`` is configurable.

The posterior mean and a 95% credible interval are stored
on BeliefProfile for later calibration analysis (e.g. ECE).
The per-belief score used for aggregation is mapped to
[-1, 1] via ``2*posterior_mean - 1`` so it stays comparable
to the baseline scorer's range.
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from scipy.stats import beta as beta_dist

from belief_stability.models import (
    BeliefProfile,
    BeliefStabilityResult,
)
from .base import BaseBeliefScorer


class BayesianScorer(BaseBeliefScorer):

    method_name = "bayesian"

    def __init__(self, absent_discount: float = 0.5) -> None:

        self.absent_discount = absent_discount

    def _posterior(self, profile: BeliefProfile) -> tuple[float, float, float]:

        alpha = 1.0 + profile.support_weight

        beta_param = (
            1.0
            + profile.contradict_weight
            + self.absent_discount * profile.absent_weight
        )

        posterior_mean = alpha / (alpha + beta_param)

        lo, hi = beta_dist.ppf([0.025, 0.975], alpha, beta_param)

        return posterior_mean, float(lo), float(hi)

    def compute(
        self,
        profiles: List[BeliefProfile],
    ) -> BeliefStabilityResult:

        if not profiles:

            return BeliefStabilityResult(
                stability_score=0.0,
                profiles=[],
                method=self.method_name,
            )

        scores = []

        for profile in profiles:

            posterior_mean, lo, hi = self._posterior(profile)

            profile.posterior_mean = posterior_mean
            profile.credible_interval = (lo, hi)

            belief_score = 2.0 * posterior_mean - 1.0

            profile.score = belief_score

            scores.append(belief_score)

        stability_score = sum(scores) / len(scores)

        return BeliefStabilityResult(
            stability_score=stability_score,
            profiles=profiles,
            method=self.method_name,
        )
