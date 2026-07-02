"""
---------------------------------------------------------
Belief Stability Module

Canonicalizer

Main orchestration pipeline.
---------------------------------------------------------
"""

from __future__ import annotations

from belief_stability.models import (
    Belief,
    ExtractedClaim,
)

from .attribute_mapper import AttributeMapper
from .builder import BeliefBuilder
from .entity_mapper import EntityMapper
from .preprocessing import Preprocessor
from .relation_mapper import RelationMapper


class Canonicalizer:

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

        # Step 1
        claim = self.preprocessor.preprocess(claim)

        # Step 2
        claim = self.relation_mapper.normalize(claim)

        # Step 3
        claim = self.entity_mapper.normalize(claim)

        # Step 4
        claim = self.attribute_mapper.normalize(claim)

        # Step 5
        return self.builder.build(claim)