"""
---------------------------------------------------------
Belief Stability Module

File        : pronoun_resolver.py

Description
-----------
Deterministic, NER-gated pronoun resolution for WikiBio-style
biographical text.

WikiBio's GPT-3 samples restate the document subject's full
name at the start of the passage (a structural property of
how the samples were generated - confirmed empirically), so
the subject can be identified once per document from the
first original sentence, then reused to resolve pronouns in
both the original sentences and every sampled passage of
that same document.

This is intentionally conservative: if a second named person
is introduced in the current or immediately preceding
sentence, the sentence is left untouched rather than risking
a wrong substitution (e.g. "He married Jane Smith. She was a
novelist." - resolving "She" to the document subject would
misattribute the second sentence). No neural coreference
model is used - see project discussion for why a heuristic
is preferred at this stage.
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

from belief_stability.nlp import get_nlp

THIRD_PERSON_PRONOUNS = {"he", "him", "his", "she", "her", "hers"}


def identify_primary_subject(first_sentence: str) -> Optional[str]:
    """
    Identify the document's primary subject from its first
    original sentence. Returns None (resolution skipped
    entirely) if no PERSON entity is found - safer than
    guessing.
    """

    if not first_sentence:
        return None

    doc = get_nlp()(first_sentence)

    for ent in doc.ents:

        if ent.label_ == "PERSON":
            return ent.text.strip()

    return None


def _surname(name: str) -> str:

    tokens = name.strip().split()

    return tokens[-1].lower() if tokens else ""


def _same_person(entity_text: str, primary_subject: str, primary_surname: str) -> bool:

    entity_text = entity_text.strip()

    entity_lower = entity_text.lower()

    primary_lower = primary_subject.lower()

    if entity_lower == primary_lower:
        return True

    if entity_lower in primary_lower or primary_lower in entity_lower:
        return True

    return _surname(entity_text) == primary_surname


def _replace_pronouns(doc, primary_subject: str) -> str:

    pieces: List[str] = []

    for token in doc:

        if token.pos_ == "PRON" and token.lower_ in THIRD_PERSON_PRONOUNS:
            pieces.append(primary_subject + token.whitespace_)
        else:
            pieces.append(token.text_with_ws)

    return "".join(pieces).strip()


def resolve_pronouns(sentences: List[str], primary_subject: Optional[str]) -> List[str]:
    """
    Resolve third-person singular pronouns (he/him/his/she/her/
    hers) to ``primary_subject`` wherever no competing named
    person is present in the current or previous sentence.
    Plural pronouns (they/them/their) are deliberately left
    alone - in this dataset they more often refer to
    institutions/groups than to the document subject.
    """

    if not primary_subject or not sentences:
        return list(sentences)

    nlp = get_nlp()

    docs = [nlp(sentence) for sentence in sentences]

    primary_surname = _surname(primary_subject)

    resolved: List[str] = []

    for i in range(len(docs)):

        competing = False

        for j in (i - 1, i):

            if j < 0:
                continue

            for ent in docs[j].ents:

                if ent.label_ == "PERSON" and not _same_person(
                    ent.text, primary_subject, primary_surname
                ):
                    competing = True
                    break

            if competing:
                break

        if competing:
            resolved.append(sentences[i])
        else:
            resolved.append(_replace_pronouns(docs[i], primary_subject))

    return resolved
