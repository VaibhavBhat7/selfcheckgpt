from belief_stability.canonicalizer.document_entity_normalizer import DocumentEntityNormalizer
from belief_stability.models import Belief


def test_title_stripping_always_applied():

    normalizer = DocumentEntityNormalizer()

    alias_map = normalizer.build_alias_map(
        primary_subject="John Reynolds",
        all_strings=["Dr. John Reynolds", "John Reynolds"],
    )

    assert alias_map["Dr. John Reynolds"] == "John Reynolds"


def test_last_name_merge_when_unambiguous():

    normalizer = DocumentEntityNormalizer()

    alias_map = normalizer.build_alias_map(
        primary_subject="John Reynolds",
        all_strings=["John Reynolds", "Reynolds", "lawyer"],
    )

    assert alias_map["Reynolds"] == "John Reynolds"


def test_last_name_merge_blocked_when_second_person_shares_surname():

    normalizer = DocumentEntityNormalizer()

    alias_map = normalizer.build_alias_map(
        primary_subject="John Reynolds",
        all_strings=["John Reynolds", "Jane Reynolds", "Reynolds", "lawyer"],
    )

    assert "Reynolds" not in alias_map


def test_nickname_merge_matches_primary_first_name():

    normalizer = DocumentEntityNormalizer()

    alias_map = normalizer.build_alias_map(
        primary_subject="William Shakespeare",
        all_strings=["William Shakespeare", "Bill"],
    )

    assert alias_map["Bill"] == "William Shakespeare"


def test_nickname_merge_skipped_when_first_name_does_not_match():

    normalizer = DocumentEntityNormalizer()

    alias_map = normalizer.build_alias_map(
        primary_subject="John Reynolds",
        all_strings=["John Reynolds", "Bill"],
    )

    assert "Bill" not in alias_map


def test_no_primary_subject_returns_empty_map():

    normalizer = DocumentEntityNormalizer()

    alias_map = normalizer.build_alias_map(primary_subject=None, all_strings=["Reynolds"])

    assert alias_map == {}


def test_apply_rewrites_subject_and_object():

    normalizer = DocumentEntityNormalizer()

    beliefs = [
        Belief(belief_id="1", subject="Reynolds", relation="OCCUPATION", object="lawyer"),
    ]

    alias_map = {"Reynolds": "John Reynolds"}

    rewritten = normalizer.apply(beliefs, alias_map)

    assert rewritten[0].subject == "John Reynolds"
    assert rewritten[0].object == "lawyer"


def test_apply_with_empty_alias_map_returns_same_list():

    normalizer = DocumentEntityNormalizer()

    beliefs = [Belief(belief_id="1", subject="X", relation="R", object="Y")]

    assert normalizer.apply(beliefs, {}) is beliefs
