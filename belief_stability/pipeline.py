"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Stability Pipeline

File        : pipeline.py

Description
-----------
End-to-end orchestration pipeline for the Belief Stability
module.

Pipeline
--------
Extracted Claims
        ↓
Canonicalization
        ↓
Belief Matching
        ↓
Belief Persistence
        ↓
Belief Stability Score

This pipeline assumes claim extraction has already been
performed (e.g., using REBEL).
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from belief_stability.canonicalizer import Canonicalizer
from belief_stability.matcher import BeliefMatcher
from belief_stability.models import (
    ExtractedClaim,
    PassageBeliefs,
    TransitionResult,
    BeliefStabilityResult,
)
from belief_stability.scoring import (
    BeliefPersistence,
    BeliefScorer,
)


class BeliefStabilityPipeline:
    """
    End-to-end Belief Stability pipeline.
    """

    def __init__(
        self,
        canonicalizer: Canonicalizer | None = None,
        matcher: BeliefMatcher | None = None,
        persistence: BeliefPersistence | None = None,
        scorer: BeliefScorer | None = None,
    ) -> None:

        self.canonicalizer = canonicalizer or Canonicalizer()
        self.matcher = matcher or BeliefMatcher()
        self.persistence = persistence or BeliefPersistence()
        self.scorer = scorer or BeliefScorer()

    # -----------------------------------------------------
    # Internal Helper
    # -----------------------------------------------------

    def _canonicalize_passage(
        self,
        passage_id: int,
        claims: List[ExtractedClaim],
    ) -> PassageBeliefs:
        """
        Convert extracted claims into canonical beliefs.
        """

        beliefs = [
            self.canonicalizer.canonicalize(claim)
            for claim in claims
        ]

        return PassageBeliefs(
            passage_id=passage_id,
            beliefs=beliefs,
        )

    # -----------------------------------------------------
    # Main Pipeline
    # -----------------------------------------------------

    def run(
        self,
        original_claims: List[ExtractedClaim],
        sampled_claims: List[List[ExtractedClaim]],
    ) -> BeliefStabilityResult:
        """
        Execute the complete Belief Stability pipeline.

        Parameters
        ----------
        original_claims : List[ExtractedClaim]
            Claims extracted from the original passage.

        sampled_claims : List[List[ExtractedClaim]]
            Claims extracted from each sampled passage.

        Returns
        -------
        BeliefStabilityResult
        """

        # ---------------------------------------------
        # Canonicalize Original Passage
        # ---------------------------------------------

        original_beliefs = self._canonicalize_passage(
            passage_id=0,
            claims=original_claims,
        )

        # ---------------------------------------------
        # Canonicalize Sampled Passages
        # ---------------------------------------------

        sampled_passages = [

            self._canonicalize_passage(
                passage_id=index + 1,
                claims=claims,
            )

            for index, claims in enumerate(sampled_claims)

        ]

        # ---------------------------------------------
        # Belief Matching
        # ---------------------------------------------

        transition_results: List[TransitionResult] = []

        for sampled_passage in sampled_passages:

            transition_results.extend(

                self.matcher.match_all(
                    original_beliefs,
                    sampled_passage,
                )

            )

        # ---------------------------------------------
        # Persistence Analysis
        # ---------------------------------------------

        profiles = self.persistence.compute(
            transition_results
        )

        # ---------------------------------------------
        # Final Scoring
        # ---------------------------------------------

        return self.scorer.compute(
            profiles
        )