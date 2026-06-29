"""
---------------------------------------------------------
Belief Stability Module

Entity Mapper

Description
-----------
Performs lightweight entity normalization.

This module is intentionally simple and deterministic.
---------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

from belief_stability.constants import ENTITY_ALIASES
from belief_stability.models import ExtractedClaim


class EntityMapper:
    """
    Performs lightweight entity normalization.
    """

    @staticmethod
    def _normalize_entity(entity: str) -> str:

        if not entity:
            return ""

        entity = entity.strip()

        lookup_key = entity.lower()

        return ENTITY_ALIASES.get(
            lookup_key,
            entity
        )

    def normalize(
        self,
        claim: ExtractedClaim,
    ) -> ExtractedClaim:
        """
        Normalize subject and object entities.
        """

        return replace(
            claim,
            subject=self._normalize_entity(claim.subject),
            object=self._normalize_entity(claim.object),
        )