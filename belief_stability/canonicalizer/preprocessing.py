"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Canonicalizer

File        : preprocessing.py

Description:
    Performs lightweight preprocessing on extracted claims
    before canonicalization.

    IMPORTANT:
    This module performs ONLY syntactic cleaning.

    It DOES NOT:
        - normalize relations
        - normalize entities
        - infer semantics
        - extract attributes

---------------------------------------------------------
"""

from __future__ import annotations

import re
import unicodedata

from belief_stability.models import ExtractedClaim


class Preprocessor:
    """
    Lightweight preprocessing for extracted claims.

    This class performs only surface-level text cleaning.
    No semantic transformations are performed.
    """

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean a single text field.

        Operations
        ----------
        1. Unicode normalization
        2. Trim leading/trailing whitespace
        3. Collapse multiple spaces
        4. Normalize quotation marks

        Parameters
        ----------
        text : str

        Returns
        -------
        str
            Cleaned text.
        """

        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Normalize quotation marks
        text = (
            text.replace("“", '"')
                .replace("”", '"')
                .replace("‘", "'")
                .replace("’", "'")
        )

        # Remove extra whitespace
        text = text.strip()

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text

    def preprocess(self, claim: ExtractedClaim) -> ExtractedClaim:
        """
        Preprocess an extracted claim.

        Parameters
        ----------
        claim : ExtractedClaim

        Returns
        -------
        ExtractedClaim
            Cleaned claim.
        """

        return ExtractedClaim(
            subject=self._clean_text(claim.subject),
            relation=self._clean_text(claim.relation),
            object=self._clean_text(claim.object),
            confidence=claim.confidence,
        )