"""
---------------------------------------------------------
Belief Stability Module

Belief Builder

Description
-----------
Converts a normalized ExtractedClaim into a canonical
Belief object.

This class performs no normalization or reasoning.
---------------------------------------------------------
"""

from __future__ import annotations

import uuid

from belief_stability.models import (
    Belief,
    ExtractedClaim,
)


class BeliefBuilder:
    """
    Builds canonical Belief objects.
    """

    def build(
        self,
        claim: ExtractedClaim
    ) -> Belief:
        """
        Convert a normalized ExtractedClaim into a Belief.

        Parameters
        ----------
        claim : ExtractedClaim

        Returns
        -------
        Belief
        """

        return Belief(

            belief_id=str(uuid.uuid4()),

            subject=claim.subject,

            relation=claim.relation,

            object=claim.object,

            attributes=claim.attributes,

            confidence=claim.confidence,

            source_text=None,
        )