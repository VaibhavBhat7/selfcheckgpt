"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework

Description
-----------
Belief Stability is a claim-centric framework for measuring
the persistence of factual beliefs across multiple
stochastic generations of a Large Language Model (LLM).

Pipeline
--------
Raw Passage
        ↓
Claim Extraction
        ↓
Canonicalization
        ↓
Belief Matching
        ↓
Persistence Analysis
        ↓
Belief Stability Score
---------------------------------------------------------
"""

from .models import (
    ExtractedClaim,
    Belief,
    PassageBeliefs,
    Transition,
    TransitionResult,
    BeliefProfile,
    BeliefStabilityResult,
)

from .claim_extractor import (
    BaseClaimExtractor,
    RebelClaimExtractor,
)

from .canonicalizer import (
    Canonicalizer,
)

from .matcher import (
    BeliefLookup,
    TransitionClassifier,
    BeliefMatcher,
)

from .scoring import (
    BeliefPersistence,
    BeliefScorer,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "ExtractedClaim",
    "Belief",
    "PassageBeliefs",
    "Transition",
    "TransitionResult",
    "BeliefProfile",
    "BeliefStabilityResult",

    # Claim Extraction
    "BaseClaimExtractor",
    "RebelClaimExtractor",

    # Canonicalization
    "Canonicalizer",

    # Matching
    "BeliefLookup",
    "TransitionClassifier",
    "BeliefMatcher",

    # Scoring
    "BeliefPersistence",
    "BeliefScorer",
]