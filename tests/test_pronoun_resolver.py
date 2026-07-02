from belief_stability.pronoun_resolver import identify_primary_subject, resolve_pronouns


def test_identify_primary_subject_returns_first_person():

    subject = identify_primary_subject("John Russell Reynolds was an English lawyer.")

    assert subject is not None
    assert "Reynolds" in subject


def test_identify_primary_subject_returns_none_when_no_person():

    subject = identify_primary_subject("The weather was pleasant that day.")

    assert subject is None


def test_resolve_pronouns_replaces_unambiguous_pronoun():

    sentences = [
        "John Reynolds was born in London.",
        "He was a lawyer.",
    ]

    resolved = resolve_pronouns(sentences, "John Reynolds")

    assert len(resolved) == 2
    assert "He" not in resolved[1]
    assert "John Reynolds" in resolved[1]


def test_resolve_pronouns_blocks_on_competing_named_person():

    sentences = [
        "John Reynolds married Jane Smith in 1990.",
        "She was a novelist.",
    ]

    resolved = resolve_pronouns(sentences, "John Reynolds")

    # "Jane Smith" is a competing named person in the previous sentence -
    # resolution must abstain rather than misattribute "She" to Reynolds.
    assert resolved[1] == sentences[1]


def test_resolve_pronouns_no_primary_subject_returns_sentences_unchanged():

    sentences = ["He was a lawyer."]

    resolved = resolve_pronouns(sentences, None)

    assert resolved == sentences


def test_resolve_pronouns_empty_sentences_returns_empty():

    assert resolve_pronouns([], "John Reynolds") == []
