"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Matching

File        : transition_classifier.py

Description
-----------
Classifies the transition of an original belief against
candidate beliefs retrieved from a sampled passage, using
a tiered decision process:

Tier 1 (EXACT)    - candidate.object == original.object,
                    same attributes. Free, always on.
Tier INVERSE      - a candidate matched via an inverse or
                    symmetric relation lookup (e.g. (A,
                    FOUNDED, B) vs (B, FOUNDED_BY, A)) is a
                    complete structural match by construction
                    - no similarity scoring needed. Checked
                    right after Tier 1, before Tier 2.
Tier 2 (SEMANTIC) - cosine similarity of candidate.object
                    vs original.object embeddings exceeds
                    ``match_similarity_threshold``. Catches
                    paraphrases ("Apple Inc" vs "Apple").
                    Requires a SemanticMatcher with a
                    prebuilt embedding cache - no model
                    runs online.
Tier 3 (NLI)      - only for a narrow ambiguous similarity
                    band, reuses the DeBERTa-MNLI checkpoint
                    SelfCheckGPT-NLI already uses. Off by
                    default (config-gated) since it is the
                    only tier that touches a model online.

Possible transition states:
    - SUPPORT
    - ABSENT
    - CONTRADICT
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

from belief_stability.models import (
    Belief,
    MatchTier,
    Transition,
    TransitionResult,
)


class TransitionClassifier:
    """
    Classifies belief transitions using tiered matching.
    """

    def __init__(
        self,
        semantic_matcher=None,
        nli_arbitrator=None,
        match_similarity_threshold: float = 0.82,
        nli_ambiguous_low: float = 0.55,
        nli_ambiguous_high: float = 0.82,
    ) -> None:

        self.semantic_matcher = semantic_matcher

        self.nli_arbitrator = nli_arbitrator

        self.match_similarity_threshold = match_similarity_threshold

        self.nli_ambiguous_low = nli_ambiguous_low

        self.nli_ambiguous_high = nli_ambiguous_high

    # --------------------------------------------------

    @staticmethod
    def _exact_match(
        original_belief: Belief,
        candidate: Belief,
    ) -> bool:

        return (
            candidate.object == original_belief.object
            and candidate.attributes == original_belief.attributes
        )

    # --------------------------------------------------

    def _best_semantic_candidate(
        self,
        original_belief: Belief,
        candidate_beliefs: List[Belief],
    ) -> Optional[tuple[Belief, float]]:

        if self.semantic_matcher is None:
            return None

        best_candidate = None

        best_score = -1.0

        for candidate in candidate_beliefs:

            score = self.semantic_matcher.similarity(
                original_belief.object,
                candidate.object,
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return None

        return best_candidate, best_score

    # --------------------------------------------------

    def classify(
        self,
        original_belief: Belief,
        candidate_beliefs: List[Belief],
        inverse_candidates: Optional[List[Belief]] = None,
    ) -> TransitionResult:
        """
        Classify the transition of an original belief.

        Parameters
        ----------
        original_belief : Belief
            Belief extracted from the original passage.

        candidate_beliefs : List[Belief]
            Candidate beliefs retrieved from the sampled passage
            sharing (subject, relation) with ``original_belief``.

        inverse_candidates : Optional[List[Belief]]
            Candidates retrieved via an inverse/symmetric
            relation lookup (see BeliefLookup.lookup_inverse) -
            already structurally verified, so any non-empty
            list here is an immediate SUPPORT.

        Returns
        -------
        TransitionResult
        """

        inverse_candidates = inverse_candidates or []

        # -----------------------------------------
        # No candidate belief found (direct or inverse) -> ABSENT
        # -----------------------------------------

        if not candidate_beliefs and not inverse_candidates:

            return TransitionResult(
                original_belief=original_belief,
                transition=Transition.ABSENT,
                matched_belief=None,
                tier=MatchTier.NONE,
                match_score=0.0,
            )

        # -----------------------------------------
        # Tier 1: exact match
        # -----------------------------------------

        for candidate in candidate_beliefs:

            if self._exact_match(original_belief, candidate):

                return TransitionResult(
                    original_belief=original_belief,
                    transition=Transition.SUPPORT,
                    matched_belief=candidate,
                    tier=MatchTier.EXACT,
                    match_score=1.0,
                )

        # -----------------------------------------
        # Tier INVERSE: structural inverse/symmetric match
        # -----------------------------------------

        if inverse_candidates:

            return TransitionResult(
                original_belief=original_belief,
                transition=Transition.SUPPORT,
                matched_belief=inverse_candidates[0],
                tier=MatchTier.INVERSE,
                match_score=1.0,
            )

        # -----------------------------------------
        # Tier 2: semantic match
        # -----------------------------------------

        semantic_result = self._best_semantic_candidate(
            original_belief, candidate_beliefs
        )

        if semantic_result is not None:

            best_candidate, best_score = semantic_result

            if best_score >= self.match_similarity_threshold:

                return TransitionResult(
                    original_belief=original_belief,
                    transition=Transition.SUPPORT,
                    matched_belief=best_candidate,
                    tier=MatchTier.SEMANTIC,
                    match_score=best_score,
                )

            # -------------------------------------
            # Tier 3: NLI arbitration (optional)
            # -------------------------------------

            if (
                self.nli_arbitrator is not None
                and self.nli_ambiguous_low <= best_score < self.nli_ambiguous_high
            ):

                contradiction_prob = self.nli_arbitrator.contradiction_probability(
                    original_belief, best_candidate
                )

                if contradiction_prob < 0.5:

                    return TransitionResult(
                        original_belief=original_belief,
                        transition=Transition.SUPPORT,
                        matched_belief=best_candidate,
                        tier=MatchTier.NLI,
                        match_score=1.0 - contradiction_prob,
                    )

                return TransitionResult(
                    original_belief=original_belief,
                    transition=Transition.CONTRADICT,
                    matched_belief=best_candidate,
                    tier=MatchTier.NLI,
                    match_score=contradiction_prob,
                )

        # -----------------------------------------
        # Contradiction (candidates exist for the same
        # subject/relation, but none support the belief)
        # -----------------------------------------

        return TransitionResult(
            original_belief=original_belief,
            transition=Transition.CONTRADICT,
            matched_belief=candidate_beliefs[0],
            tier=MatchTier.EXACT,
            match_score=1.0,
        )
