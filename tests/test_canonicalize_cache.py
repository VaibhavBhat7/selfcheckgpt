from pathlib import Path

from belief_stability.belief_cache import ClaimCache, ExampleClaims
from belief_stability.canonicalize_cache import CacheCanonicalizer
from belief_stability.config import BeliefStabilityConfig
from belief_stability.models import ExtractedClaim


class FakeSemanticMatcher:
    """No-op stand-in - avoids downloading MiniLM in unit tests."""

    def __init__(self):
        self.embedding_cache = {}

    def build_cache(self, strings):
        self.embedding_cache = {s: None for s in strings}
        return self.embedding_cache

    def save(self, file_path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_bytes(b"")


def test_canonicalize_example_maps_relation_and_entity():

    example = ExampleClaims(
        dataset_id=1,
        primary_subject=None,
        original_claims=[
            [ExtractedClaim(subject="Steve Jobs", relation="founder of", object="Apple")],
        ],
        sampled_claims=[],
    )

    canonicalizer = CacheCanonicalizer()

    beliefs = canonicalizer.canonicalize_example(example)

    belief = beliefs.original_beliefs[0][0]

    assert belief.relation == "FOUNDED"
    assert belief.subject == "Steve Jobs"


def test_document_entity_normalization_merges_unambiguous_last_name():

    example = ExampleClaims(
        dataset_id=1,
        primary_subject="John Reynolds",
        original_claims=[
            [ExtractedClaim(subject="John Reynolds", relation="occupation", object="lawyer")],
        ],
        sampled_claims=[
            [ExtractedClaim(subject="Reynolds", relation="occupation", object="lawyer")],
        ],
    )

    canonicalizer = CacheCanonicalizer()

    beliefs = canonicalizer.canonicalize_example(example)

    sampled_subject = beliefs.sampled_beliefs[0][0].subject

    assert sampled_subject == "John Reynolds"


def test_document_entity_normalization_blocks_ambiguous_surname():

    example = ExampleClaims(
        dataset_id=1,
        primary_subject="John Reynolds",
        original_claims=[
            [ExtractedClaim(subject="John Reynolds", relation="spouse", object="Jane Reynolds")],
        ],
        sampled_claims=[
            [ExtractedClaim(subject="Reynolds", relation="occupation", object="lawyer")],
        ],
    )

    canonicalizer = CacheCanonicalizer()

    beliefs = canonicalizer.canonicalize_example(example)

    # "Reynolds" alone is ambiguous (a second Reynolds - Jane - exists in
    # this document), so the merge must NOT happen.
    sampled_subject = beliefs.sampled_beliefs[0][0].subject

    assert sampled_subject == "Reynolds"


def test_ablation_flag_disables_document_entity_normalization():

    example = ExampleClaims(
        dataset_id=1,
        primary_subject="John Reynolds",
        original_claims=[
            [ExtractedClaim(subject="John Reynolds", relation="occupation", object="lawyer")],
        ],
        sampled_claims=[
            [ExtractedClaim(subject="Reynolds", relation="occupation", object="lawyer")],
        ],
    )

    config = BeliefStabilityConfig(use_document_entity_normalization=False)

    canonicalizer = CacheCanonicalizer(config=config)

    beliefs = canonicalizer.canonicalize_example(example)

    sampled_subject = beliefs.sampled_beliefs[0][0].subject

    assert sampled_subject == "Reynolds"


def test_canonicalize_cache_builds_belief_cache_and_embeddings(tmp_path):

    claim_cache = ClaimCache(entries={
        1: ExampleClaims(
            dataset_id=1,
            primary_subject="Alice Smith",
            original_claims=[
                [ExtractedClaim(subject="Alice Smith", relation="occupation", object="physicist")],
            ],
            sampled_claims=[
                [ExtractedClaim(subject="Alice Smith", relation="occupation", object="physicist")],
            ],
        ),
    })

    canonicalizer = CacheCanonicalizer(semantic_matcher=FakeSemanticMatcher())

    belief_cache = canonicalizer.canonicalize_cache(claim_cache)

    assert len(belief_cache) == 1
    assert belief_cache.get(1).original_beliefs[0][0].relation == "OCCUPATION"

    output_path = tmp_path / "belief_cache.pkl"

    belief_cache.save(output_path)

    canonicalizer.build_embedding_cache(belief_cache, output_path)

    embeddings_path = output_path.with_suffix(output_path.suffix + ".embeddings.pkl")

    assert embeddings_path.exists()
