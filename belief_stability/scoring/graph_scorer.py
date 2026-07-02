"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Scoring

File        : graph_scorer.py

Description
-----------
Flagship novel scoring arm: belief-graph consistency
propagation. ``profiles`` passed into ``compute()`` are all
the beliefs extracted from one original passage, so this
builds one graph per passage - no extra passage-boundary
bookkeeping needed.

Nodes  : beliefs of the original passage.
Edges  : two beliefs are connected if they share a
         canonicalized subject or object entity string
         (e.g. two beliefs both about "Steve Jobs").
Signal : each node's base score is the Bayesian posterior
         score (BayesianScorer). One-hop propagation then
         smooths each node toward its neighbors' scores:

    score'_i = (1 - alpha) * score_i + alpha * mean(neighbor scores)

A belief contradicted across sampled passages pulls down
confidence in other beliefs about the same entity within
the same passage - the closest formulation to the "Belief
Stability" framing in the project title, and the headline
ablation arm against SelfCheckGPT.
---------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from belief_stability.models import (
    BeliefProfile,
    BeliefStabilityResult,
)
from .base import BaseBeliefScorer
from .bayesian_scorer import BayesianScorer


class GraphScorer(BaseBeliefScorer):

    method_name = "graph"

    def __init__(
        self,
        alpha: float = 0.3,
        absent_discount: float = 0.5,
        base_scorer: BayesianScorer | None = None,
    ) -> None:

        self.alpha = alpha

        self.base_scorer = base_scorer or BayesianScorer(
            absent_discount=absent_discount
        )

    # --------------------------------------------------

    @staticmethod
    def _build_adjacency(
        profiles: List[BeliefProfile],
    ) -> Dict[int, List[int]]:

        entity_to_nodes: Dict[str, List[int]] = defaultdict(list)

        for i, profile in enumerate(profiles):

            for entity in (profile.belief.subject, profile.belief.object):

                if entity:
                    entity_to_nodes[entity].append(i)

        adjacency: Dict[int, List[int]] = defaultdict(list)

        for nodes in entity_to_nodes.values():

            if len(nodes) < 2:
                continue

            for i in nodes:
                for j in nodes:
                    if i != j:
                        adjacency[i].append(j)

        return adjacency

    # --------------------------------------------------

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

        # Base per-belief scores via the Bayesian posterior.
        base_scores = []

        for profile in profiles:

            posterior_mean, lo, hi = self.base_scorer._posterior(profile)

            profile.posterior_mean = posterior_mean
            profile.credible_interval = (lo, hi)

            base_scores.append(2.0 * posterior_mean - 1.0)

        adjacency = self._build_adjacency(profiles)

        propagated_scores = []

        for i, profile in enumerate(profiles):

            neighbors = adjacency.get(i, [])

            if not neighbors:
                propagated = base_scores[i]

            else:
                neighbor_mean = sum(
                    base_scores[j] for j in neighbors
                ) / len(neighbors)

                propagated = (
                    (1.0 - self.alpha) * base_scores[i]
                    + self.alpha * neighbor_mean
                )

            profile.graph_score = propagated

            profile.score = propagated

            propagated_scores.append(propagated)

        stability_score = sum(propagated_scores) / len(propagated_scores)

        return BeliefStabilityResult(
            stability_score=stability_score,
            profiles=profiles,
            method=self.method_name,
        )
