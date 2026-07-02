from belief_stability.matcher.lookup import BeliefLookup
from belief_stability.matcher.matcher import BeliefMatcher
from belief_stability.matcher.transition_classifier import TransitionClassifier
from belief_stability.models import Belief, MatchTier, PassageBeliefs, Transition


def _belief(subject, relation, obj):
    return Belief(belief_id=f"{subject}-{relation}-{obj}", subject=subject, relation=relation, object=obj)


def test_lookup_inverse_finds_role_swapped_inverse_pair():

    original = _belief("Steve Jobs", "FOUNDED", "Apple")

    candidate = _belief("Apple", "FOUNDED_BY", "Steve Jobs")

    lookup = BeliefLookup()

    index = lookup.build_index(PassageBeliefs(passage_id=1, beliefs=[candidate]))

    results = lookup.lookup_inverse(original, index)

    assert results == [candidate]


def test_lookup_inverse_finds_symmetric_relation():

    original = _belief("Alice", "SPOUSE", "Bob")

    candidate = _belief("Bob", "SPOUSE", "Alice")

    lookup = BeliefLookup()

    index = lookup.build_index(PassageBeliefs(passage_id=1, beliefs=[candidate]))

    results = lookup.lookup_inverse(original, index)

    assert results == [candidate]


def test_lookup_inverse_returns_empty_for_relations_without_inverse():

    original = _belief("Alice", "OCCUPATION", "physicist")

    candidate = _belief("physicist", "OCCUPATION", "Alice")

    lookup = BeliefLookup()

    index = lookup.build_index(PassageBeliefs(passage_id=1, beliefs=[candidate]))

    results = lookup.lookup_inverse(original, index)

    assert results == []


def test_lookup_inverse_requires_full_structural_roundtrip():

    original = _belief("Steve Jobs", "FOUNDED", "Apple")

    # Object doesn't round-trip back to the original subject - not a match.
    candidate = _belief("Apple", "FOUNDED_BY", "Someone Else")

    lookup = BeliefLookup()

    index = lookup.build_index(PassageBeliefs(passage_id=1, beliefs=[candidate]))

    results = lookup.lookup_inverse(original, index)

    assert results == []


def test_transition_classifier_returns_inverse_tier_without_semantic_matcher():

    original = _belief("Steve Jobs", "FOUNDED", "Apple")

    inverse_candidate = _belief("Apple", "FOUNDED_BY", "Steve Jobs")

    # No semantic_matcher configured - if this reached Tier 2 it would
    # crash or silently no-op; getting SUPPORT/INVERSE here proves the
    # inverse tier short-circuits before semantic matching.
    classifier = TransitionClassifier()

    result = classifier.classify(original, candidate_beliefs=[], inverse_candidates=[inverse_candidate])

    assert result.transition == Transition.SUPPORT
    assert result.tier == MatchTier.INVERSE
    assert result.match_score == 1.0


def test_transition_classifier_absent_when_no_direct_or_inverse_candidates():

    original = _belief("Steve Jobs", "FOUNDED", "Apple")

    classifier = TransitionClassifier()

    result = classifier.classify(original, candidate_beliefs=[], inverse_candidates=[])

    assert result.transition == Transition.ABSENT
    assert result.tier == MatchTier.NONE


def test_belief_matcher_end_to_end_inverse_support():

    original_belief = _belief("Steve Jobs", "FOUNDED", "Apple")

    candidate = _belief("Apple", "FOUNDED_BY", "Steve Jobs")

    matcher = BeliefMatcher()

    result = matcher.match(
        original_belief,
        PassageBeliefs(passage_id=1, beliefs=[candidate]),
    )

    assert result.transition == Transition.SUPPORT
    assert result.tier == MatchTier.INVERSE


def test_belief_matcher_respects_use_inverse_matching_flag():

    original_belief = _belief("Steve Jobs", "FOUNDED", "Apple")

    candidate = _belief("Apple", "FOUNDED_BY", "Steve Jobs")

    matcher = BeliefMatcher(use_inverse_matching=False)

    result = matcher.match(
        original_belief,
        PassageBeliefs(passage_id=1, beliefs=[candidate]),
    )

    assert result.transition == Transition.ABSENT
