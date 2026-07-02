from belief_stability.belief_cache import ClaimCache, RawClaimCacheBuilder
from belief_stability.claim_extractor import ClaimExtractor
from belief_stability.models import ExtractedClaim


class FakeExtractor(ClaimExtractor):
    """Deterministic stand-in for RebelClaimExtractor - no model download."""

    def __init__(self):
        self.calls = 0

    def extract_batch(self, texts):

        self.calls += 1

        return [
            [ExtractedClaim(subject="X", relation="rel", object=text[:5])]
            if text.strip() else []
            for text in texts
        ]


FAKE_DATASET = [
    {
        "wiki_bio_test_idx": 1,
        "gpt3_sentences": ["Alice was born in Paris.", "Alice studied physics."],
        "gpt3_text_samples": ["Sample one text.", "Sample two text."],
    },
    {
        "wiki_bio_test_idx": 2,
        "gpt3_sentences": ["Bob was born in Rome."],
        "gpt3_text_samples": ["Sample three text.", "Sample four text."],
    },
]


def build(tmp_path, dataset=FAKE_DATASET, flush_every=10, resume=True, extractor=None):

    extractor = extractor or FakeExtractor()

    builder = RawClaimCacheBuilder(extractor=extractor)

    output_path = tmp_path / "claim_cache.pkl"

    cache = builder.build(
        dataset=dataset,
        output_path=output_path,
        flush_every=flush_every,
        resume=resume,
    )

    return cache, output_path, extractor


def test_build_produces_one_entry_per_example(tmp_path):

    cache, output_path, extractor = build(tmp_path)

    assert len(cache) == 2
    assert cache.has(1) and cache.has(2)

    entry = cache.get(1)

    assert len(entry.original_claims) == 2  # two original sentences
    assert len(entry.sampled_claims) == 2   # two sampled passages (one sentence each)

    # Each sentence/passage produced exactly one extracted claim.
    assert all(len(c) == 1 for c in entry.original_claims)
    assert all(len(c) == 1 for c in entry.sampled_claims)


def test_save_load_roundtrip(tmp_path):

    cache, output_path, _ = build(tmp_path)

    assert output_path.exists()

    loaded = ClaimCache.load(output_path)

    assert len(loaded) == len(cache)
    assert loaded.get(1).original_claims[0][0].subject == "X"


def test_resume_skips_already_cached_examples(tmp_path):

    extractor = FakeExtractor()

    cache, output_path, _ = build(tmp_path, extractor=extractor)

    calls_after_first_build = extractor.calls

    # Re-run with resume=True: nothing new to extract.
    second_cache, _, _ = build(tmp_path, extractor=extractor, resume=True)

    assert extractor.calls == calls_after_first_build
    assert len(second_cache) == 2
