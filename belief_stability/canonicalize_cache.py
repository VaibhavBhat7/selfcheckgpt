"""
---------------------------------------------------------
Belief Stability Module

File        : canonicalize_cache.py

Description
-----------
Second (fast, CPU-only) offline stage: turns a raw
``ClaimCache`` (belief_cache.py) into a canonicalized
``BeliefCache``.

Per document:

1. Every raw ``ExtractedClaim`` is run through the existing
   per-claim ``Canonicalizer`` (preprocessing, relation
   mapping, global entity aliasing, attribute mapping) -
   unchanged internals, just an expanded RELATION_MAPPING.
2. The document's full pool of resulting subject/object
   strings is normalized ONCE via ``DocumentEntityNormalizer``
   (title stripping, ambiguity-gated last-name/nickname
   merging), then that alias map is applied to every belief
   in the document (originals and samples alike, so matching
   later sees a consistent entity space).

This whole stage re-runs in low minutes over the full
238-document dataset (spaCy NER over small per-document
string pools, no REBEL) - cheap enough to re-run any time
canonicalization/matching config changes, which is the whole
point of splitting it out from the raw-claims cache.
---------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from belief_stability.belief_cache import BeliefCache, ClaimCache, ExampleBeliefs, ExampleClaims
from belief_stability.canonicalizer import Canonicalizer, DocumentEntityNormalizer
from belief_stability.config import BeliefStabilityConfig
from belief_stability.matcher.semantic import SemanticMatcher
from belief_stability.models import Belief, ExtractedClaim
from belief_stability.utils import setup_logger

logger = setup_logger(__name__)


class CacheCanonicalizer:

    def __init__(
        self,
        config: BeliefStabilityConfig | None = None,
        canonicalizer: Canonicalizer | None = None,
        document_normalizer: DocumentEntityNormalizer | None = None,
        semantic_matcher: SemanticMatcher | None = None,
    ) -> None:

        self.config = config or BeliefStabilityConfig()

        self.canonicalizer = canonicalizer or Canonicalizer()

        self.document_normalizer = document_normalizer or DocumentEntityNormalizer()

        self.semantic_matcher = semantic_matcher or SemanticMatcher()

    # --------------------------------------------------

    def _canonicalize_claim_lists(
        self,
        claim_lists: List[List[ExtractedClaim]],
    ) -> List[List[Belief]]:

        return [
            [self.canonicalizer.canonicalize(claim) for claim in claims]
            for claims in claim_lists
        ]

    # --------------------------------------------------

    def canonicalize_example(self, example: ExampleClaims) -> ExampleBeliefs:

        original_beliefs = self._canonicalize_claim_lists(example.original_claims)

        sampled_beliefs = self._canonicalize_claim_lists(example.sampled_claims)

        if self.config.use_document_entity_normalization and example.primary_subject:

            pool: List[str] = []

            for beliefs in (*original_beliefs, *sampled_beliefs):
                for belief in beliefs:
                    pool.append(belief.subject)
                    pool.append(belief.object)

            alias_map = self.document_normalizer.build_alias_map(
                example.primary_subject, pool
            )

            original_beliefs = [
                self.document_normalizer.apply(beliefs, alias_map)
                for beliefs in original_beliefs
            ]

            sampled_beliefs = [
                self.document_normalizer.apply(beliefs, alias_map)
                for beliefs in sampled_beliefs
            ]

        return ExampleBeliefs(
            dataset_id=example.dataset_id,
            original_beliefs=original_beliefs,
            sampled_beliefs=sampled_beliefs,
        )

    # --------------------------------------------------

    def canonicalize_cache(
        self,
        claim_cache: ClaimCache,
        limit: int | None = None,
    ) -> BeliefCache:

        belief_cache = BeliefCache()

        dataset_ids = list(claim_cache.entries.keys())

        if limit is not None:
            dataset_ids = dataset_ids[:limit]

        for dataset_id in dataset_ids:

            example = claim_cache.get(dataset_id)

            belief_cache.entries[dataset_id] = self.canonicalize_example(example)

        logger.info(
            "Canonicalized %s examples (document_entity_normalization=%s, "
            "inverse_matching=%s).",
            len(belief_cache), self.config.use_document_entity_normalization,
            self.config.use_inverse_matching,
        )

        return belief_cache

    # --------------------------------------------------

    def build_embedding_cache(
        self,
        belief_cache: BeliefCache,
        output_path: str | Path,
    ) -> None:
        """
        Batch-embed every distinct subject/object string seen
        across the whole (final, canonicalized) cache, once, so
        online semantic matching never runs the embedding model.
        Moved here (from the old BeliefCacheBuilder) because
        canonicalization changes the strings being embedded.
        """

        output_path = Path(output_path)

        strings = set()

        for entry in belief_cache.entries.values():

            for beliefs in (*entry.original_beliefs, *entry.sampled_beliefs):

                for belief in beliefs:
                    strings.add(belief.subject)
                    strings.add(belief.object)

        self.semantic_matcher.build_cache(strings)

        embeddings_path = output_path.with_suffix(output_path.suffix + ".embeddings.pkl")

        self.semantic_matcher.save(embeddings_path)

        logger.info(
            "Embedding cache built: %s distinct strings -> %s",
            len(strings), embeddings_path,
        )
