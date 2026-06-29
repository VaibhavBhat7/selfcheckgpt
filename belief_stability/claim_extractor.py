"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Claim Extraction

File        : claim_extractor.py

Description
-----------
Wrapper around the REBEL claim extraction model.

The extractor converts raw passages into structured
ExtractedClaim objects.

This class is intentionally model-agnostic so that REBEL
can later be replaced by another extractor without
modifying the downstream Belief Stability pipeline.

---------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from belief_stability.models import ExtractedClaim


class BaseClaimExtractor(ABC):
    """
    Abstract interface for all claim extractors.
    """

    @abstractmethod
    def extract(
        self,
        text: str,
    ) -> List[ExtractedClaim]:
        """
        Extract structured claims from text.

        Parameters
        ----------
        text : str

        Returns
        -------
        List[ExtractedClaim]
        """
        raise NotImplementedError


class RebelClaimExtractor(BaseClaimExtractor):
    """
    Wrapper around the REBEL model.

    NOTE
    ----
    This class currently serves as a wrapper interface.

    The actual REBEL model loading and inference will be
    implemented here.
    """

    def __init__(self) -> None:
        """
        Initialize the REBEL model.

        TODO
        ----
        Load tokenizer and model.
        """

        pass

    def extract(
        self,
        text: str,
    ) -> List[ExtractedClaim]:
        """
        Extract claims using REBEL.

        Parameters
        ----------
        text : str

        Returns
        -------
        List[ExtractedClaim]

        TODO
        ----
        Replace placeholder implementation with actual
        REBEL inference.
        """

        raise NotImplementedError(
            "REBEL extraction has not been implemented yet."
        )