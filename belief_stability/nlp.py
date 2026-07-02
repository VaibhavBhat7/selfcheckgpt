"""
---------------------------------------------------------
Belief Stability Module

File        : nlp.py

Description
-----------
Lazily loads and caches a single spaCy ``en_core_web_sm``
pipeline. Sentence splitting, pronoun resolution, and
document-scoped entity normalization all need spaCy NER -
this makes sure the model is loaded once per process rather
than once per caller.
---------------------------------------------------------
"""

from __future__ import annotations

import spacy

_nlp = None


def get_nlp():

    global _nlp

    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")

    return _nlp
