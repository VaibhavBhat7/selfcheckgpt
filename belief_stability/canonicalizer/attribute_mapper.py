"""
---------------------------------------------------------
Belief Stability Module

Attribute Mapper

Description
-----------
Normalizes attribute values.

This module performs normalization only.
It never extracts new attributes.
---------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

from belief_stability.models import ExtractedClaim


class AttributeMapper:
    """
    Normalizes attribute values.
    """

    def normalize(
        self,
        claim: ExtractedClaim,
    ) -> ExtractedClaim:
        """
        Normalize all attribute values.
        """

        normalized_attributes = {}

        for key, value in claim.attributes.items():

            if value is None:
                continue

            normalized_attributes[key] = str(value).strip()

        return replace(
            claim,
            attributes=normalized_attributes,
        )