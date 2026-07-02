from belief_stability.matcher import BeliefLookup, TransitionClassifier
from belief_stability.models import Belief, MatchTier, Transition


def make_belief(subject="Steve Jobs", relation="FOUNDED", object_="Apple", belief_id="b1"):
    return Belief(belief_id=belief_id, subject=subject, relation=relation, object=object_)


class StubSemanticMatcher:
    """Deterministic similarity stub - no model download needed."""

    def __init__(self, similarity_by_pair):
        self.similarity_by_pair = similarity_by_pair

    def similarity(self, a, b):
        return self.similarity_by_pair.get((a, b), self.similarity_by_pair.get((b, a), 0.0))


def test_absent_when_no_candidates():

    classifier = TransitionClassifier()

    result = classifier.classify(make_belief(), candidate_beliefs=[])

    assert result.transition == Transition.ABSENT
    assert result.tier == MatchTier.NONE


def test_exact_match_supports():

    classifier = TransitionClassifier()

    original = make_belief(object_="Apple")
    candidate = make_belief(object_="Apple", belief_id="b2")

    result = classifier.classify(original, candidate_beliefs=[candidate])

    assert result.transition == Transition.SUPPORT
    assert result.tier == MatchTier.EXACT
    assert result.match_score == 1.0


def test_no_exact_match_without_semantic_matcher_contradicts():

    classifier = TransitionClassifier()

    original = make_belief(object_="Apple")
    candidate = make_belief(object_="Pixar", belief_id="b2")

    result = classifier.classify(original, candidate_beliefs=[candidate])

    assert result.transition == Transition.CONTRADICT


def test_semantic_match_above_threshold_supports():

    original = make_belief(object_="Apple")
    candidate = make_belief(object_="Apple Inc", belief_id="b2")

    matcher = StubSemanticMatcher({("Apple", "Apple Inc"): 0.9})

    classifier = TransitionClassifier(
        semantic_matcher=matcher, match_similarity_threshold=0.82
    )

    result = classifier.classify(original, candidate_beliefs=[candidate])

    assert result.transition == Transition.SUPPORT
    assert result.tier == MatchTier.SEMANTIC
    assert result.match_score == 0.9


def test_semantic_match_below_threshold_contradicts():

    original = make_belief(object_="Apple")
    candidate = make_belief(object_="Pixar", belief_id="b2")

    matcher = StubSemanticMatcher({("Apple", "Pixar"): 0.1})

    classifier = TransitionClassifier(
        semantic_matcher=matcher, match_similarity_threshold=0.82
    )

    result = classifier.classify(original, candidate_beliefs=[candidate])

    assert result.transition == Transition.CONTRADICT


def test_lookup_index_groups_by_subject_relation():

    lookup = BeliefLookup()

    from belief_stability.models import PassageBeliefs

    passage = PassageBeliefs(
        passage_id=1,
        beliefs=[
            make_belief(object_="Apple", belief_id="b1"),
            make_belief(object_="Pixar", belief_id="b2"),
            make_belief(subject="Tim Cook", object_="Apple", belief_id="b3"),
        ],
    )

    index = lookup.build_index(passage)

    assert len(index[("Steve Jobs", "FOUNDED")]) == 2
    assert len(index[("Tim Cook", "FOUNDED")]) == 1
