"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Canonicalizer

File        : canonicalizer.py

Description
-----------
Main orchestration pipeline for canonicalizing extracted
claims into canonical Belief objects.

This class coordinates all canonicalization components but
contains no normalization logic itself.
---------------------------------------------------------
"""

from __future__ import annotations

from belief_stability.models import Belief, ExtractedClaim

from .attribute_mapper import AttributeMapper
from .builder import BeliefBuilder
from .entity_mapper import EntityMapper
from .preprocessing import Preprocessor
from .relation_mapper import RelationMapper


class Canonicalizer:
    """
    Main canonicalization pipeline.

    Pipeline
    --------
    ExtractedClaim
            ↓
    Preprocessing
            ↓
    Relation Mapping
            ↓
    Entity Mapping
            ↓
    Attribute Mapping
            ↓
    Belief Construction
            ↓
    Belief
    """

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        relation_mapper: RelationMapper | None = None,
        entity_mapper: EntityMapper | None = None,
        attribute_mapper: AttributeMapper | None = None,
        builder: BeliefBuilder | None = None,
    ) -> None:

        self.preprocessor = preprocessor or Preprocessor()
        self.relation_mapper = relation_mapper or RelationMapper()
        self.entity_mapper = entity_mapper or EntityMapper()
        self.attribute_mapper = attribute_mapper or AttributeMapper()
        self.builder = builder or BeliefBuilder()

    def canonicalize(
        self,
        claim: ExtractedClaim,
    ) -> Belief:
        """
        Convert an ExtractedClaim into a canonical Belief.

        Parameters
        ----------
        claim : ExtractedClaim

        Returns
        -------
        Belief
            Canonical belief representation.
        """

        # --------------------------------------------
        # Step 1 : Preprocessing
        # --------------------------------------------

        claim = self.preprocessor.preprocess(claim)

        # --------------------------------------------
        # Step 2 : Relation Normalization
        # --------------------------------------------

        claim.relation = self.relation_mapper.normalize(
            claim.relation
        )

        # --------------------------------------------
        # Step 3 : Entity Normalization
        # --------------------------------------------

        claim.subject = self.entity_mapper.normalize(
            claim.subject
        )

        claim.object = self.entity_mapper.normalize(
            claim.object
        )

        # --------------------------------------------
        # Step 4 : Attribute Normalization
        # --------------------------------------------

        claim.attributes = self.attribute_mapper.normalize(
            claim.attributes
        )

        # --------------------------------------------
        # Step 5 : Build Canonical Belief
        # --------------------------------------------

        return self.builder.build(claim)