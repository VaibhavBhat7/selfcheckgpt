"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Stability

Description:
    Core data models used throughout the Belief Stability
    module. Every downstream component operates on these
    dataclasses rather than raw dictionaries or JSON.

---------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Transition(Enum):
    """
    Possible transition states of an original belief
    across sampled passages.
    """

    SUPPORT = "support"
    ABSENT = "absent"
    CONTRADICT = "contradict"


class MatchTier(Enum):
    """
    Which stage of the tiered belief matcher produced a
    transition decision.
    """

    EXACT = "exact"
    INVERSE = "inverse"
    SEMANTIC = "semantic"
    NLI = "nli"
    NONE = "none"

@dataclass
class ExtractedClaim:

    subject: str

    relation: str

    object: str

    attributes: Dict[str, str] = field(default_factory=dict)

    confidence: float = 1.0

@dataclass(slots=True)
class Belief:
    """
    Canonical representation of a factual belief.

    Every extracted claim is converted into this format
    before belief matching.

    Example
    -------
    Belief(
        belief_id="b001",
        subject="Steve Jobs",
        relation="FOUNDED",
        object="Apple",
        attributes={"year": "1976"}
    )
    """

    belief_id: str

    subject: str

    relation: str

    object: str

    attributes: Dict[str, str] = field(default_factory=dict)

    source_text: Optional[str] = None

    confidence: float = 1.0


@dataclass(slots=True)
class PassageBeliefs:
    """
    Collection of beliefs extracted from one passage.

    A passage may contain multiple factual claims.

    Example
    -------
    Original Passage
        ↓
    12 beliefs

    Sample Passage
        ↓
    15 beliefs
    """

    passage_id: int

    beliefs: List[Belief] = field(default_factory=list)


@dataclass(slots=True)
class TransitionResult:
    """
    Stores the transition outcome for one original belief
    against one sampled passage.
    """

    original_belief: Belief

    transition: Transition

    matched_belief: Optional[Belief] = None

    tier: MatchTier = MatchTier.NONE

    match_score: float = 0.0


@dataclass(slots=True)
class BeliefProfile:
    """
    Transition statistics for one belief across all sampled
    passages.

    ``support``/``absent``/``contradict`` are raw transition
    counts (used by the baseline scorer). The ``*_weight``
    fields are the same transitions summed by match-tier
    confidence (``TransitionResult.match_score``), used by
    the Bayesian and graph scorers so a low-confidence
    semantic match counts as weaker evidence than an exact
    match.

    Example
    -------
    Support      : 16
    Absent       : 3
    Contradict   : 1
    """

    belief: Belief

    support: int = 0

    absent: int = 0

    contradict: int = 0

    support_weight: float = 0.0

    absent_weight: float = 0.0

    contradict_weight: float = 0.0

    posterior_mean: Optional[float] = None

    credible_interval: Optional[Tuple[float, float]] = None

    graph_score: Optional[float] = None

    # Generic per-belief scalar in [-1, 1], set by whichever scorer
    # produced this profile (baseline's (S-C)/Total, Bayesian's
    # 2*posterior_mean-1, or graph's propagated score). Lets callers
    # re-aggregate belief -> sentence with something other than the
    # scorer's default mean (e.g. min) without caring which scorer
    # was used - see scoring/base.py::aggregate.
    score: Optional[float] = None


@dataclass(slots=True)
class BeliefStabilityResult:
    """
    Final output of the Belief Stability module.

    Stores the overall stability score together with
    individual belief profiles.
    """

    stability_score: float

    profiles: List[BeliefProfile] = field(default_factory=list)

    method: str = "baseline"