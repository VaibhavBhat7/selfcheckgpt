import pytest

from belief_stability.config import BeliefStabilityConfig
from belief_stability.models import ExtractedClaim
from belief_stability.pipeline import BeliefStabilityPipeline

# Fixture mirrors experiments/test_bs_pipeline.py's Steve Jobs example:
#   founder_of Apple : SUPPORT, CONTRADICT (Pixar), SUPPORT -> across 3 samples
#   occupation        : SUPPORT, ABSENT,             CONTRADICT (Inventor)

ORIGINAL_CLAIMS = [
    ExtractedClaim(subject="Steve Jobs", relation="founder_of", object="Apple"),
    ExtractedClaim(subject="Steve Jobs", relation="occupation", object="Entrepreneur"),
]

SAMPLE_1 = [
    ExtractedClaim(subject="Steve Jobs", relation="founder_of", object="Apple"),
    ExtractedClaim(subject="Steve Jobs", relation="occupation", object="Entrepreneur"),
]

SAMPLE_2 = [
    ExtractedClaim(subject="Steve Jobs", relation="founder_of", object="Pixar"),
]

SAMPLE_3 = [
    ExtractedClaim(subject="Steve Jobs", relation="founder_of", object="Apple"),
    ExtractedClaim(subject="Steve Jobs", relation="occupation", object="Inventor"),
]

SAMPLED_CLAIMS = [SAMPLE_1, SAMPLE_2, SAMPLE_3]


def run_with_method(method: str):

    config = BeliefStabilityConfig(scoring_method=method, use_semantic_matching=False)

    pipeline = BeliefStabilityPipeline(config=config)

    return pipeline.run(
        original_claims=ORIGINAL_CLAIMS,
        sampled_claims=SAMPLED_CLAIMS,
    )


def test_baseline_scorer_matches_v1_formula():

    result = run_with_method("baseline")

    assert result.method == "baseline"
    # founder_of Apple: support=2, contradict=1, absent=0 -> (2-1)/3
    # occupation:       support=1, absent=1, contradict=1 -> (1-1)/3 = 0
    assert result.stability_score == pytest.approx((1 / 3 + 0.0) / 2, abs=1e-6)


def test_bayesian_scorer_produces_calibrated_posterior():

    result = run_with_method("bayesian")

    assert result.method == "bayesian"

    for profile in result.profiles:
        assert profile.posterior_mean is not None
        assert 0.0 <= profile.posterior_mean <= 1.0
        lo, hi = profile.credible_interval
        assert lo <= profile.posterior_mean <= hi

    assert -1.0 <= result.stability_score <= 1.0


def test_graph_scorer_propagates_between_shared_entity_beliefs():

    result = run_with_method("graph")

    assert result.method == "graph"
    assert len(result.profiles) == 2

    # Both beliefs share subject "Steve Jobs", so with alpha > 0 each
    # profile's propagated graph_score should differ from a pure,
    # unpropagated Bayesian score derived from the same posterior mean.
    for profile in result.profiles:
        assert profile.graph_score is not None
        unpropagated = 2.0 * profile.posterior_mean - 1.0
        assert profile.graph_score != pytest.approx(unpropagated)


def test_empty_profiles_score_zero():

    config = BeliefStabilityConfig(scoring_method="baseline")

    pipeline = BeliefStabilityPipeline(config=config)

    result = pipeline.run(original_claims=[], sampled_claims=[[]])

    assert result.stability_score == 0.0
    assert result.profiles == []
