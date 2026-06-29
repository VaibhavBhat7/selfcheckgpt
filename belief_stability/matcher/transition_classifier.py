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
candidate beliefs retrieved from a sampled passage.

Possible transition states:
    - SUPPORT
    - ABSENT
    - CONTRADICT
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from belief_stability.models import (
    Belief,
    Transition,
    TransitionResult,
)


class TransitionClassifier:
    """
    Classifies belief transitions.
    """

    def classify(
        self,
        original_belief: Belief,
        candidate_beliefs: List[Belief],
    ) -> TransitionResult:
        """
        Classify the transition of an original belief.

        Parameters
        ----------
        original_belief : Belief
            Belief extracted from the original passage.

        candidate_beliefs : List[Belief]
            Candidate beliefs retrieved from the sampled passage.

        Returns
        -------
        TransitionResult
        """

        # -----------------------------------------
        # No candidate belief found
        # -----------------------------------------

        if not candidate_beliefs:

            return TransitionResult(
                original_belief=original_belief,
                transition=Transition.ABSENT,
                matched_belief=None,
            )

        # -----------------------------------------
        # Search for supporting belief
        # -----------------------------------------

        for candidate in candidate_beliefs:

            if (
                candidate.object == original_belief.object
                and candidate.attributes == original_belief.attributes
            ):

                return TransitionResult(
                    original_belief=original_belief,
                    transition=Transition.SUPPORT,
                    matched_belief=candidate,
                )

        # -----------------------------------------
        # Contradiction
        # -----------------------------------------

        return TransitionResult(
            original_belief=original_belief,
            transition=Transition.CONTRADICT,
            matched_belief=candidate_beliefs[0],
        )
