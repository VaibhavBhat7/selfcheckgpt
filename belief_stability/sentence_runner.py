"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Sentence Runner

File        : sentence_runner.py

Description
-----------
Adapter between SelfCheckGPT and the Belief Stability
pipeline.

Two entry points, deliberately kept separate:

``run_from_cache`` (primary, batch-eval path)
    Takes already-extracted, already-canonicalized Belief
    lists (e.g. from a BeliefCache built by
    experiments/build_belief_cache.py). No extractor or
    model is ever loaded - pure Python matching/scoring.
    This is what evaluate_belief.py and the WikiBio
    notebook should use; it is what makes re-running an
    evaluation (e.g. to compare scoring methods) take
    seconds instead of hours.

``run_sentence`` / ``run_sentences`` (legacy raw-text path)
    Extracts claims from raw text on the fly via REBEL.
    Kept for one-off/interactive use (e.g. scoring a single
    new sentence you just typed) - NOT for batch dataset
    evaluation, since it re-extracts every sampled passage
    for every sentence it is called with.
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from belief_stability.claim_extractor import ClaimExtractor, RebelClaimExtractor
from belief_stability.models import Belief
from belief_stability.pipeline import BeliefStabilityPipeline
from belief_stability.scoring import aggregate as aggregate_result


class SentenceBeliefRunner:
    """
    Adapter class used to integrate Belief Stability
    into SelfCheckGPT.
    """

    def __init__(
        self,
        extractor: ClaimExtractor | None = None,
        pipeline: BeliefStabilityPipeline | None = None,
        lazy_extractor: bool = True,
    ) -> None:

        self._extractor = extractor

        self._lazy_extractor = lazy_extractor and extractor is None

        self.pipeline = pipeline or BeliefStabilityPipeline()

        if not self._lazy_extractor and self._extractor is None:
            self._extractor = RebelClaimExtractor()

    # --------------------------------------------------

    @property
    def extractor(self) -> ClaimExtractor:
        """
        Lazily instantiate REBEL only if/when the raw-text
        path is actually used, so run_from_cache never
        loads a model.
        """

        if self._extractor is None:
            self._extractor = RebelClaimExtractor()

        return self._extractor

    # --------------------------------------------------
    # Primary path: precomputed beliefs, GPU-free.
    # --------------------------------------------------

    def run_from_cache(
        self,
        original_beliefs: List[List[Belief]],
        sampled_beliefs: List[List[Belief]],
        aggregation: str = "mean",
    ) -> List[float]:
        """
        Compute Belief Stability scores for multiple
        original sentences using already-extracted,
        already-canonicalized beliefs.

        Parameters
        ----------
        original_beliefs : List[List[Belief]]
            One belief list per original sentence.

        sampled_beliefs : List[List[Belief]]
            One belief list per sampled passage (extracted
            once per passage, shared across every sentence).

        aggregation : str
            How a sentence's own (possibly multiple) belief
            scores are combined into one sentence score:
            "mean" (the scorer's default) or "min"
            (pessimistic - the worst belief drags the
            sentence down). See scoring/base.py::aggregate.

        Returns
        -------
        List[float]
            One Belief Stability score per original sentence.
        """

        scores: List[float] = []

        for sentence_beliefs in original_beliefs:

            result = self.pipeline.run_from_beliefs(
                original_beliefs=sentence_beliefs,
                sampled_beliefs=sampled_beliefs,
            )

            if aggregation == "mean":
                scores.append(result.stability_score)
            else:
                scores.append(aggregate_result(result, method=aggregation))

        return scores

    # --------------------------------------------------
    # Legacy path: raw text, extracts on the fly.
    # --------------------------------------------------

    def run_sentence(
        self,
        original_sentence: str,
        sampled_passages: List[str],
    ) -> float:
        """
        Compute the Belief Stability score for one
        original sentence.

        Parameters
        ----------
        original_sentence : str

        sampled_passages : List[str]

        Returns
        -------
        float
        """

        if original_sentence is None:
            return 0.0

        original_sentence = original_sentence.strip()

        if len(original_sentence) == 0:
            return 0.0

        original_claims = self.extractor.extract(
            original_sentence
        )

        sampled_claims = self.extractor.extract_batch(
            sampled_passages
        )

        result = self.pipeline.run(
            original_claims=original_claims,
            sampled_claims=sampled_claims,
        )

        return result.stability_score

    # --------------------------------------------------

    def run_sentences(
        self,
        original_sentences: List[str],
        sampled_passages: List[str],
    ) -> List[float]:
        """
        Compute Belief Stability scores for multiple
        original sentences from raw text.

        WARNING: this is the slow path - it re-extracts all
        sampled passages once per call (i.e. once per
        original sentence, since each call re-extracts).
        Prefer building a BeliefCache and using
        run_from_cache for batch dataset evaluation.

        Parameters
        ----------
        original_sentences : List[str]
            Sentences from the original passage.

        sampled_passages : List[str]
            Multiple sampled passages generated by the LLM.

        Returns
        -------
        List[float]
            One Belief Stability score per original sentence.
        """

        scores: List[float] = []

        for sentence in original_sentences:

            score = self.run_sentence(
                original_sentence=sentence,
                sampled_passages=sampled_passages,
            )

            scores.append(score)

        return scores

    # --------------------------------------------------

    def __call__(
        self,
        original_sentences: List[str],
        sampled_passages: List[str],
    ) -> List[float]:
        """
        Convenience wrapper.

        Allows usage like:

        runner(
            original_sentences,
            sampled_passages,
        )
        """

        return self.run_sentences(
            original_sentences=original_sentences,
            sampled_passages=sampled_passages,
        )
