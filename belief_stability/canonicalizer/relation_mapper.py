"""
---------------------------------------------------------
Belief Stability Module

Relation Mapper

Description
-----------
Maps extractor-specific relation labels to the canonical
relation vocabulary.

This module performs lexical normalization only.
---------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import replace

from belief_stability.constants import RELATION_MAPPING
from belief_stability.models import ExtractedClaim


class RelationMapper:
    """
    Maps raw relation labels into canonical relations.
    """

    @staticmethod
    def _normalize_format(relation: str) -> str:
        """
        Normalize formatting without changing semantics.
        """

        if not relation:
            return ""

        relation = relation.strip().lower()

        relation = relation.replace("-", "_")
        relation = relation.replace(" ", "_")

        relation = re.sub(r"_+", "_", relation)

        return relation

    def normalize(
        self,
        claim: ExtractedClaim,
    ) -> ExtractedClaim:
        """
        Normalize the relation of an extracted claim.
        """

        relation = self._normalize_format(claim.relation)

        canonical_relation = RELATION_MAPPING.get(
            relation,
            relation.upper()
        )

        return replace(
            claim,
            relation=canonical_relation
        )