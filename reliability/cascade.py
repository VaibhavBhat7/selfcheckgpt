"""
---------------------------------------------------------
Reliability Aggregation Engine

File        : cascade.py

Description
-----------
Production Reliability Cascade - replaces the single global
logistic regression fusion (aggregator.py::ReliabilityAggregator)
as the primary reliability scoring strategy.

Motivation (experimental evidence, see experiments/compare_fusion_strategies.py
and experiments/diagnose_fusion.py): a single global LR fit across
the whole dataset implies a decision boundary on selfcheck_score far
above the 0.5 convention used to evaluate it, which mechanically
under-recalls the consistent-hallucination subgroup regardless of
whether Belief Stability's signal is any good. A multi-stage cascade
- trust SelfCheckGPT when it's already confident, only consult a
secondary signal when it isn't - dominated the LR fusion on BOTH
aggregate AUC-PR and subgroup recall in that comparison.

Architecture
------------
    Input
      |
      v
    SelfCheckGPT
      |
      +-- confident (score >= its own ROC-optimal threshold)
      |       -> return SelfCheckGPT's score immediately
      |
      +-- not confident
              |
              v
          Belief Stability
              |
              +-- (future: confident -> return; else escalate further)
              +-- for now: always resolves (terminal stage)
                      |
                      v
              [Counterfactual Verification - not yet implemented]
                      |
                      v
              [Adversarial Verification - not yet implemented]
                      |
                      v
              Final Reliability Score

Each stage is a ``CascadeStage`` that decides, per row, whether it
can resolve a final score with sufficient confidence or whether the
row should escalate to the next stage. Adding Counterfactual /
Adversarial Verification later is: implement a new ``CascadeStage``
subclass and append it to the ``stages`` list passed to
``ReliabilityCascade`` - nothing else in this file, or in
``experiments/evaluate_reliability.py``, needs to change.

All stage-fitting parameters (the SelfCheckGPT threshold, the Belief
Stability graph alpha) must be computed from TRAINING-fold data only
by the caller and passed in - this module does no cross-validation
itself, matching the leak-safety pattern already used throughout
reliability/ (see subgroups.py, alpha_selection.py).
---------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .features import DocumentContext, SentenceRow, attach_graph_scores


@dataclass
class CascadeStageOutput:

    score: float

    resolved: bool

    stage: str


class CascadeStage(ABC):
    """
    One stage of the reliability cascade. ``predict`` receives only
    the rows that reached this stage (i.e. were NOT resolved by any
    earlier stage) and must return one ``CascadeStageOutput`` per
    row, in the same order.
    """

    name: str = "base"

    @abstractmethod
    def predict(self, rows: Sequence[SentenceRow]) -> List[CascadeStageOutput]:
        raise NotImplementedError


class SelfCheckGPTStage(CascadeStage):
    """
    Resolves a row immediately (trusts SelfCheckGPT's own score) if
    that score is at or above ``threshold`` - the fold's ROC-optimal
    (Youden's J) decision boundary, i.e. exactly the same threshold
    used to define the consistent-hallucination subgroup
    (see reliability/subgroups.py::compute_subgroup_threshold). Rows
    below the threshold - SelfCheckGPT's own blind spot - escalate.
    """

    name = "selfcheckgpt"

    def __init__(self, threshold: float) -> None:

        self.threshold = threshold

    def predict(self, rows: Sequence[SentenceRow]) -> List[CascadeStageOutput]:

        return [
            CascadeStageOutput(
                score=row.selfcheck_score,
                resolved=row.selfcheck_score >= self.threshold,
                stage=self.name,
            )
            for row in rows
        ]


class BeliefStabilityStage(CascadeStage):
    """
    Blends SelfCheckGPT's score with the Belief Stability graph
    score for rows that escalated past SelfCheckGPTStage. Belief
    Stability's score is on [-1, 1] (supported..contradicted) -
    converted to a [0, 1] "risk of non-factual" scale via
    ``(1 - score) / 2`` before blending, so it's directly comparable
    to selfcheck_score.

    ``is_terminal=True`` (the current default, since no downstream
    stage exists yet) means every row this stage sees is resolved -
    it always returns a final score rather than escalating further.
    Once Counterfactual Verification exists, a confidence-based gate
    can be added here (analogous to SelfCheckGPTStage's threshold)
    and ``is_terminal`` set to False, without touching anything else
    in the cascade.
    """

    name = "belief_stability"

    def __init__(
        self,
        contexts: Dict[int, DocumentContext],
        alpha: float,
        absent_discount: float = 0.5,
        blend_weight: float = 0.5,
        is_terminal: bool = True,
    ) -> None:

        self.contexts = contexts
        self.alpha = alpha
        self.absent_discount = absent_discount
        self.blend_weight = blend_weight
        self.is_terminal = is_terminal

    @staticmethod
    def _risk(bs_score: float) -> float:

        return (1.0 - bs_score) / 2.0

    def predict(self, rows: Sequence[SentenceRow]) -> List[CascadeStageOutput]:

        if not rows:
            return []

        graph_scores = attach_graph_scores(rows, self.contexts, self.alpha, self.absent_discount)

        results = []

        for row in rows:

            bs_score = graph_scores[(row.dataset_id, row.sentence_index)]

            blended = (
                self.blend_weight * row.selfcheck_score
                + (1.0 - self.blend_weight) * self._risk(bs_score)
            )

            results.append(CascadeStageOutput(score=blended, resolved=self.is_terminal, stage=self.name))

        return results


class ReliabilityCascade:
    """
    Orchestrates an ordered list of ``CascadeStage`` instances. Each
    row is offered to stages in order; the first stage that resolves
    it supplies the final score. The LAST stage in ``stages`` must
    resolve every row it receives (enforced at ``predict()`` time) so
    every input row always gets a final score.
    """

    def __init__(self, stages: Sequence[CascadeStage]) -> None:

        if not stages:
            raise ValueError("ReliabilityCascade requires at least one stage.")

        self.stages = list(stages)

        self.last_stage_assignment: Dict[Tuple[int, int], str] = {}

    def predict(self, rows: Sequence[SentenceRow]) -> Dict[Tuple[int, int], float]:

        remaining = list(rows)

        final_scores: Dict[Tuple[int, int], float] = {}

        stage_assignment: Dict[Tuple[int, int], str] = {}

        for stage in self.stages:

            if not remaining:
                break

            outputs = stage.predict(remaining)

            still_unresolved = []

            for row, output in zip(remaining, outputs):

                key = (row.dataset_id, row.sentence_index)

                if output.resolved:
                    final_scores[key] = output.score
                    stage_assignment[key] = output.stage
                else:
                    still_unresolved.append(row)

            remaining = still_unresolved

        if remaining:
            raise RuntimeError(
                f"{len(remaining)} row(s) were not resolved by any cascade stage - "
                "the last stage passed to ReliabilityCascade must resolve every "
                "row it receives (e.g. BeliefStabilityStage(is_terminal=True))."
            )

        self.last_stage_assignment = stage_assignment

        return final_scores
