"""
---------------------------------------------------------
Belief Stability Module

File        : sentence_splitter.py

Description
-----------
Splits a passage into sentences using the same spaCy
pipeline that produced WikiBio's own ``gpt3_sentences``
field, so sampled passages are split symmetrically with the
originals instead of by a different (and possibly weaker)
heuristic.
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from belief_stability.nlp import get_nlp


def split_into_sentences(text: str) -> List[str]:

    if not text:
        return []

    doc = get_nlp()(text)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]
