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
performed (e.g., using REBEL) - see belief_cache.py for the
offline extraction/canonicalization stage. This class only
ever does canonicalization + matching + persistence +
scoring on beliefs it is given, so it never needs a GPU.
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from belief_stability.canonicalizer import Canonicalizer
from belief_stability.config import BeliefStabilityConfig
from belief_stability.matcher import BeliefMatcher
from belief_stability.models import (
    Belief,
    ExtractedClaim,
    PassageBeliefs,
    TransitionResult,
    BeliefStabilityResult,
)
from belief_stability.scoring import (
    BaseBeliefScorer,
    BeliefPersistence,
    build_scorer,
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
        scorer: BaseBeliefScorer | None = None,
        config: BeliefStabilityConfig | None = None,
        semantic_matcher=None,
        nli_arbitrator=None,
    ) -> None:

        self.config = config or BeliefStabilityConfig()

        self.canonicalizer = canonicalizer or Canonicalizer()

        self.persistence = persistence or BeliefPersistence()

        self.matcher = matcher or BeliefMatcher(
            semantic_matcher=(
                semantic_matcher if self.config.use_semantic_matching else None
            ),
            nli_arbitrator=(
                nli_arbitrator if self.config.use_nli_arbitration else None
            ),
            match_similarity_threshold=self.config.match_similarity_threshold,
            nli_ambiguous_low=self.config.nli_ambiguous_low,
            nli_ambiguous_high=self.config.nli_ambiguous_high,
            use_inverse_matching=self.config.use_inverse_matching,
        )

        self.scorer = scorer or build_scorer(
            method=self.config.scoring_method,
            absent_discount=self.config.absent_discount,
            graph_alpha=self.config.graph_alpha,
        )

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

    @staticmethod
    def _wrap_beliefs(
        passage_id: int,
        beliefs: List[Belief],
    ) -> PassageBeliefs:

        return PassageBeliefs(
            passage_id=passage_id,
            beliefs=beliefs,
        )

    # -----------------------------------------------------
    # Main Pipeline (raw claims -> canonicalize -> score)
    # -----------------------------------------------------

    def run(
        self,
        original_claims: List[ExtractedClaim],
        sampled_claims: List[List[ExtractedClaim]],
    ) -> BeliefStabilityResult:
        """
        Execute the complete Belief Stability pipeline
        starting from raw (uncanonicalized) claims.

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

        original_beliefs = self._canonicalize_passage(
            passage_id=0,
            claims=original_claims,
        )

        sampled_passages = [

            self._canonicalize_passage(
                passage_id=index + 1,
                claims=claims,
            )

            for index, claims in enumerate(sampled_claims)

        ]

        return self._run_on_beliefs(original_beliefs, sampled_passages)

    # -----------------------------------------------------
    # Cached Pipeline (already-canonicalized beliefs -> score)
    # -----------------------------------------------------

    def run_from_beliefs(
        self,
        original_beliefs: List[Belief],
        sampled_beliefs: List[List[Belief]],
    ) -> BeliefStabilityResult:
        """
        Execute the pipeline starting from already
        canonicalized beliefs (e.g. loaded from a
        BeliefCache). Skips claim extraction and
        canonicalization entirely - this is the GPU-free
        online path.
        """

        original_passage = self._wrap_beliefs(0, original_beliefs)

        sampled_passages = [
            self._wrap_beliefs(index + 1, beliefs)
            for index, beliefs in enumerate(sampled_beliefs)
        ]

        return self._run_on_beliefs(original_passage, sampled_passages)

    # -----------------------------------------------------

    def _run_on_beliefs(
        self,
        original_beliefs: PassageBeliefs,
        sampled_passages: List[PassageBeliefs],
    ) -> BeliefStabilityResult:

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
