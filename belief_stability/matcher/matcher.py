"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Matching

File        : matcher.py

Description
-----------
Main orchestration pipeline for belief matching.

Pipeline
--------
Original Belief
        ↓
Belief Lookup
        ↓
Candidate Beliefs
        ↓
Transition Classification
        ↓
TransitionResult
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from belief_stability.models import (
    Belief,
    PassageBeliefs,
    TransitionResult,
)

from .lookup import BeliefLookup
from .transition_classifier import TransitionClassifier


class BeliefMatcher:
    """
    Orchestrates the complete belief matching pipeline.
    """

    def __init__(
        self,
        lookup: BeliefLookup | None = None,
        classifier: TransitionClassifier | None = None,
    ) -> None:

        self.lookup = lookup or BeliefLookup()
        self.classifier = classifier or TransitionClassifier()

    def match(
        self,
        original_belief: Belief,
        sampled_passage: PassageBeliefs,
    ) -> TransitionResult:
        """
        Match one original belief against one sampled passage.

        Parameters
        ----------
        original_belief : Belief
            Belief extracted from the original passage.

        sampled_passage : PassageBeliefs
            Canonical beliefs extracted from one sampled passage.

        Returns
        -------
        TransitionResult
        """

        index = self.lookup.build_index(sampled_passage)

        candidate_beliefs = self.lookup.lookup(
            original_belief,
            index,
        )

        return self.classifier.classify(
            original_belief,
            candidate_beliefs,
        )

    def match_all(
        self,
        original_beliefs: PassageBeliefs,
        sampled_passage: PassageBeliefs,
    ) -> List[TransitionResult]:
        """
        Match all beliefs from the original passage against
        one sampled passage.

        Parameters
        ----------
        original_beliefs : PassageBeliefs

        sampled_passage : PassageBeliefs

        Returns
        -------
        List[TransitionResult]
        """

        index = self.lookup.build_index(sampled_passage)

        results: List[TransitionResult] = []

        for belief in original_beliefs.beliefs:

            candidates = self.lookup.lookup(
                belief,
                index,
            )

            result = self.classifier.classify(
                belief,
                candidates,
            )

            results.append(result)

        return results