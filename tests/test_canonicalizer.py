from belief_stability.canonicalizer import Canonicalizer
from belief_stability.models import ExtractedClaim


def test_relation_normalization():

    canon = Canonicalizer()

    belief = canon.canonicalize(
        ExtractedClaim(subject="Steve Jobs", relation="founder_of", object="Apple")
    )

    assert belief.relation == "FOUNDED"


def test_entity_alias_normalization():

    canon = Canonicalizer()

    belief = canon.canonicalize(
        ExtractedClaim(subject="Elon Musk", relation="born_in", object="usa")
    )

    assert belief.object == "United States"


def test_unknown_relation_falls_back_to_uppercase():

    canon = Canonicalizer()

    belief = canon.canonicalize(
        ExtractedClaim(subject="X", relation="some new relation", object="Y")
    )

    assert belief.relation == "SOME_NEW_RELATION"


def test_whitespace_cleanup():

    canon = Canonicalizer()

    belief = canon.canonicalize(
        ExtractedClaim(subject="  Steve   Jobs  ", relation="founder_of", object=" Apple ")
    )

    assert belief.subject == "Steve Jobs"
    assert belief.object == "Apple"
