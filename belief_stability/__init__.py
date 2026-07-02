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
    MatchTier,
    TransitionResult,
    BeliefProfile,
    BeliefStabilityResult,
)

from .config import BeliefStabilityConfig

from .claim_extractor import (
    ClaimExtractor,
    RebelClaimExtractor,
)

from .belief_cache import (
    BeliefCache,
    ClaimCache,
    RawClaimCacheBuilder,
    ExampleBeliefs,
    ExampleClaims,
)

from .canonicalize_cache import CacheCanonicalizer

from .canonicalizer import (
    Canonicalizer,
    DocumentEntityNormalizer,
)

from .matcher import (
    BeliefLookup,
    TransitionClassifier,
    BeliefMatcher,
    SemanticMatcher,
    NLIArbitrator,
)

from .scoring import (
    BeliefPersistence,
    BaseBeliefScorer,
    BaselineScorer,
    BayesianScorer,
    GraphScorer,
    BeliefScorer,
    build_scorer,
)

from .pipeline import BeliefStabilityPipeline

from .sentence_runner import (
    SentenceBeliefRunner,
)

__version__ = "2.0.0"

__all__ = [
    # Models
    "ExtractedClaim",
    "Belief",
    "PassageBeliefs",
    "Transition",
    "MatchTier",
    "TransitionResult",
    "BeliefProfile",
    "BeliefStabilityResult",

    # Config
    "BeliefStabilityConfig",

    # Claim Extraction
    "ClaimExtractor",
    "RebelClaimExtractor",

    # Claim Cache (offline, raw)
    "ClaimCache",
    "RawClaimCacheBuilder",
    "ExampleClaims",

    # Belief Cache (offline, canonicalized)
    "BeliefCache",
    "ExampleBeliefs",
    "CacheCanonicalizer",

    # Canonicalization
    "Canonicalizer",
    "DocumentEntityNormalizer",

    # Matching
    "BeliefLookup",
    "TransitionClassifier",
    "BeliefMatcher",
    "SemanticMatcher",
    "NLIArbitrator",

    # Scoring
    "BeliefPersistence",
    "BaseBeliefScorer",
    "BaselineScorer",
    "BayesianScorer",
    "GraphScorer",
    "BeliefScorer",
    "build_scorer",

    # Pipeline / Runner
    "BeliefStabilityPipeline",
    "SentenceBeliefRunner",
]