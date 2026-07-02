"""
---------------------------------------------------------
Belief Stability Module

File        : document_entity_normalizer.py

Description
-----------
Document-scoped entity normalization. Unlike EntityMapper
(global, always-safe country/nationality aliases), this
operates on ONE document's full pool of subject/object
strings at a time, because merging person names is only
sound with document context (a surname alone can belong to
different people across documents, or even within one
document - a spouse or child sharing the primary subject's
surname).

Three mechanisms, in increasing order of risk, all gated by
an explicit ambiguity check before merging:

1. Title stripping (Dr./Mr./Sir/...)   - always safe.
2. Last-name -> full-name merging      - blocked if another
   distinct full name in the document shares that surname.
3. Nickname -> full-name substitution  - only fires when the
   nickname's implied first name matches the primary
   subject's first name.

A loose (capitalization-based) "looks like a person name"
check is used ALONGSIDE spaCy NER for the ambiguity scan,
deliberately biased toward false positives: a wrongly-flagged
"competing name" only blocks a safe merge (falls back to
today's ABSENT-prone behavior), it can never cause an
incorrect merge - consistent with the project's precision-
first matching philosophy.
---------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Set

from belief_stability.constants import NICKNAME_TO_FIRST_NAME, PERSON_TITLES
from belief_stability.models import Belief
from belief_stability.nlp import get_nlp


class DocumentEntityNormalizer:

    @staticmethod
    def _strip_title(entity: str) -> str:

        tokens = entity.strip().split()

        while tokens and tokens[0].strip(".").lower() in PERSON_TITLES:
            tokens = tokens[1:]

        return " ".join(tokens) if tokens else entity

    @staticmethod
    def _surname(name: str) -> str:

        tokens = name.strip().split()

        return tokens[-1].lower() if tokens else ""

    @staticmethod
    def _first_name(name: str) -> str:

        tokens = name.strip().split()

        return tokens[0].lower() if tokens else ""

    @staticmethod
    def _looks_like_person_name(entity: str) -> bool:

        tokens = entity.strip().split()

        if len(tokens) < 2:
            return False

        return all(token[:1].isupper() and token[1:].islower() for token in tokens if token.isalpha())

    def _is_person_like(self, entity: str, doc) -> bool:

        if any(ent.label_ == "PERSON" for ent in doc.ents):
            return True

        return self._looks_like_person_name(entity)

    # --------------------------------------------------

    def build_alias_map(
        self,
        primary_subject: str | None,
        all_strings: List[str],
    ) -> Dict[str, str]:
        """
        Build a raw-string -> canonical-string alias map for one
        document. Returns {} if there is no known primary
        subject (nothing to anchor merges to).
        """

        if not primary_subject:
            return {}

        distinct_strings = sorted({s for s in all_strings if s})

        alias_map: Dict[str, str] = {}

        # ---- Step 1: title stripping (always safe) ----

        stripped_forms: Dict[str, str] = {}

        for entity in distinct_strings:

            stripped = self._strip_title(entity)

            stripped_forms[entity] = stripped

            if stripped != entity:
                alias_map[entity] = stripped

        working_pool = sorted(set(stripped_forms.values()))

        primary_surname = self._surname(primary_subject)

        primary_first = self._first_name(primary_subject)

        # ---- Step 2: detect competing same-surname full names ----

        nlp = get_nlp()

        docs = list(nlp.pipe(working_pool))

        full_names_by_surname: Dict[str, Set[str]] = defaultdict(set)

        for entity, doc in zip(working_pool, docs):

            if len(entity.split()) < 2:
                continue

            if not self._is_person_like(entity, doc):
                continue

            full_names_by_surname[self._surname(entity)].add(entity)

        competing_names = {
            name
            for name in full_names_by_surname.get(primary_surname, set())
            if name.lower() != primary_subject.lower()
            and self._first_name(name) != primary_first
        }

        surname_is_ambiguous = len(competing_names) > 0

        def merge_to_primary(stripped_target: str) -> None:

            for original, stripped in stripped_forms.items():

                if stripped == stripped_target:
                    alias_map[original] = primary_subject

        # ---- Step 3: last-name-only merge (blocked if ambiguous) ----

        if not surname_is_ambiguous:

            for entity in working_pool:

                tokens = entity.split()

                if (
                    len(tokens) == 1
                    and tokens[0].lower() == primary_surname
                    and entity.lower() != primary_subject.lower()
                ):
                    merge_to_primary(entity)

        # ---- Step 4: nickname merge (scoped to primary subject only) ----

        for entity in working_pool:

            tokens = entity.split()

            if len(tokens) != 1:
                continue

            implied_first_name = NICKNAME_TO_FIRST_NAME.get(tokens[0].lower())

            if implied_first_name and implied_first_name == primary_first:
                merge_to_primary(entity)

        return alias_map

    # --------------------------------------------------

    @staticmethod
    def apply(beliefs: List[Belief], alias_map: Dict[str, str]) -> List[Belief]:

        if not alias_map:
            return beliefs

        return [
            replace(
                belief,
                subject=alias_map.get(belief.subject, belief.subject),
                object=alias_map.get(belief.object, belief.object),
            )
            for belief in beliefs
        ]
