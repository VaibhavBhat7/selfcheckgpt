"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Scoring

File        : persistence.py

Description
-----------
Computes belief persistence statistics from transition
results collected across sampled passages.

---------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from belief_stability.models import (
    BeliefProfile,
    Transition,
    TransitionResult,
)


class BeliefPersistence:
    """
    Computes transition statistics for each belief.
    """

    def compute(
        self,
        transition_results: List[TransitionResult],
    ) -> List[BeliefProfile]:
        """
        Compute persistence profiles.

        Parameters
        ----------
        transition_results : List[TransitionResult]

        Returns
        -------
        List[BeliefProfile]
        """

        profiles: Dict[str, BeliefProfile] = {}

        for result in transition_results:

            belief_id = result.original_belief.belief_id

            if belief_id not in profiles:

                profiles[belief_id] = BeliefProfile(
                    belief=result.original_belief
                )

            profile = profiles[belief_id]

            # match_score defaults to 0.0 for ABSENT (no candidate was
            # found, so there is no match confidence to weight by) and
            # is otherwise the tier's confidence (1.0 for EXACT, cosine
            # similarity for SEMANTIC, 1-P(contradict) for NLI).
            weight = result.match_score

            if result.transition == Transition.SUPPORT:
                profile.support += 1
                profile.support_weight += weight

            elif result.transition == Transition.ABSENT:
                profile.absent += 1
                profile.absent_weight += 1.0

            elif result.transition == Transition.CONTRADICT:
                profile.contradict += 1
                profile.contradict_weight += weight

        return list(profiles.values())