"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Matching

File        : lookup.py

Description
-----------
Builds an index over beliefs extracted from a sampled
passage and performs efficient belief lookup.

Index Structure
---------------
(subject, relation) -> List[Belief]

This enables O(1) average-case retrieval of candidate
beliefs during belief matching.
---------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

from belief_stability.constants import INVERSE_RELATION_PAIRS, SYMMETRIC_RELATIONS
from belief_stability.models import Belief, PassageBeliefs


class BeliefLookup:
    """
    Performs indexing and lookup of canonical beliefs.
    """

    def build_index(
        self,
        passage: PassageBeliefs,
    ) -> Dict[Tuple[str, str], List[Belief]]:
        """
        Build an index for all beliefs in a passage.

        Parameters
        ----------
        passage : PassageBeliefs

        Returns
        -------
        Dict[Tuple[str, str], List[Belief]]
            Mapping from (subject, relation) to candidate beliefs.
        """

        index: DefaultDict[Tuple[str, str], List[Belief]] = defaultdict(list)

        for belief in passage.beliefs:

            key = (belief.subject, belief.relation)

            index[key].append(belief)

        return dict(index)

    def lookup(
        self,
        belief: Belief,
        index: Dict[Tuple[str, str], List[Belief]],
    ) -> List[Belief]:
        """
        Retrieve candidate beliefs for a given belief.

        Parameters
        ----------
        belief : Belief
            Original belief.

        index : Dict[Tuple[str, str], List[Belief]]
            Lookup index generated from a sampled passage.

        Returns
        -------
        List[Belief]
            Candidate beliefs having the same
            (subject, relation).
            Returns an empty list if no candidates exist.
        """

        key = (belief.subject, belief.relation)

        return index.get(key, [])

    def lookup_inverse(
        self,
        belief: Belief,
        index: Dict[Tuple[str, str], List[Belief]],
    ) -> List[Belief]:
        """
        Retrieve candidates that assert the same fact as
        ``belief`` but phrased with subject/object roles
        swapped under an inverse or symmetric relation name
        (e.g. (A, FOUNDED, B) <-> (B, FOUNDED_BY, A), or
        (A, SPOUSE, B) <-> (B, SPOUSE, A)).

        Looked up in the SAME index passed to ``lookup`` - no
        separate index structure is needed, this is just a
        differently-keyed lookup into it.

        Parameters
        ----------
        belief : Belief
            Original belief.

        index : Dict[Tuple[str, str], List[Belief]]
            Lookup index generated from a sampled passage.

        Returns
        -------
        List[Belief]
            Candidates that structurally round-trip back to
            ``belief`` under its relation's inverse/symmetric
            counterpart. Empty if the relation has no known
            inverse/symmetric counterpart, or no candidate
            round-trips.
        """

        if belief.relation in SYMMETRIC_RELATIONS:
            inverse_relation = belief.relation
        else:
            inverse_relation = INVERSE_RELATION_PAIRS.get(belief.relation)

        if inverse_relation is None:
            return []

        key = (belief.object, inverse_relation)

        candidates = index.get(key, [])

        return [
            candidate
            for candidate in candidates
            if candidate.object == belief.subject
        ]

    @staticmethod
    def contains(
        belief: Belief,
        index: Dict[Tuple[str, str], List[Belief]],
    ) -> bool:
        """
        Check whether the lookup index contains at least one
        candidate for the given belief.

        Parameters
        ----------
        belief : Belief

        index : Dict[Tuple[str, str], List[Belief]]

        Returns
        -------
        bool
        """

        key = (belief.subject, belief.relation)

        return key in index

    @staticmethod
    def size(
        index: Dict[Tuple[str, str], List[Belief]],
    ) -> int:
        """
        Returns the number of unique (subject, relation)
        entries stored in the index.

        Parameters
        ----------
        index : Dict

        Returns
        -------
        int
        """

        return len(index)