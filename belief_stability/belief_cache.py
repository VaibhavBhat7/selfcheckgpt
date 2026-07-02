"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Claim Cache

File        : belief_cache.py

Description
-----------
Offline extraction stage. Produces a cache of RAW
(pre-canonicalization) ``ExtractedClaim`` objects, not
canonicalized ``Belief`` objects - canonicalization now
happens separately (see canonicalize_cache.py) as a fast,
CPU-only, endlessly re-runnable step. This split exists so
that changes to canonicalization/matching (entity
normalization, relation mapping, inverse matching) can be
re-evaluated in seconds without ever touching REBEL again;
only changes to extraction itself (sentence granularity,
pronoun resolution) require paying this stage's cost again.

Per document, before REBEL ever runs:

1. The document's primary subject is identified once, from
   the first original sentence (WikiBio's GPT-3 samples all
   restate the subject's full name at the start of the
   passage - a structural property of how they were
   generated, confirmed empirically - so this is reliable
   without any structured dataset field).
2. Pronouns in every original sentence AND every sampled
   passage are resolved to that subject wherever
   unambiguous (see pronoun_resolver.py).
3. Sampled passages are split into individual sentences
   (sentence_splitter.py, same spaCy splitter that produced
   the dataset's own ``gpt3_sentences``) so REBEL runs at
   sentence granularity on both sides symmetrically, instead
   of on whole multi-sentence passages.

REBEL's per-sentence outputs for one sampled passage are
concatenated back into a single flat claim list for that
passage (``ExampleClaims.sampled_claims`` stays one entry
per sample, matching the shape callers already expect).

Texts are still batched in windows of ``flush_every``
dataset examples for the same reasons as before: large
padded batches for GPU efficiency, and early/resumable
checkpointing.

Pipeline
--------
Dataset
    ↓
identify_primary_subject() + resolve_pronouns() + split_into_sentences()
    ↓
flatten texts in windows of `flush_every` examples
    ↓
RebelClaimExtractor.extract_batch() (batched, fp16) -- UNCHANGED
    ↓
Claim Cache (.pkl, checkpointed/resumable)
---------------------------------------------------------
"""

from __future__ import annotations

import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from belief_stability.claim_extractor import ClaimExtractor
from belief_stability.models import Belief, ExtractedClaim
from belief_stability.pronoun_resolver import identify_primary_subject, resolve_pronouns
from belief_stability.sentence_splitter import split_into_sentences
from belief_stability.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ExampleClaims:

    dataset_id: int

    primary_subject: Optional[str] = None

    original_claims: List[List[ExtractedClaim]] = field(default_factory=list)

    sampled_claims: List[List[ExtractedClaim]] = field(default_factory=list)


class ClaimCache:
    """
    Read-only accessor over a built raw-claims cache. Used by
    canonicalize_cache.py - no model is ever loaded by this
    class.
    """

    def __init__(self, entries: Dict[int, ExampleClaims] | None = None) -> None:

        self.entries: Dict[int, ExampleClaims] = entries or {}

    def has(self, dataset_id: int) -> bool:
        return dataset_id in self.entries

    def get(self, dataset_id: int) -> ExampleClaims:

        if dataset_id not in self.entries:
            raise KeyError(f"Dataset id {dataset_id} not found in claim cache.")

        return self.entries[dataset_id]

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, file_path: str | Path) -> None:

        file_path = Path(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: str | Path) -> "ClaimCache":

        file_path = Path(file_path)

        with open(file_path, "rb") as f:
            entries = pickle.load(f)

        logger.info("Loaded claim cache: %s examples from %s", len(entries), file_path)

        return cls(entries=entries)


class RawClaimCacheBuilder:
    """
    Builds a ClaimCache from a WikiBio-style dataset.

    Each dataset example ``x`` is expected to expose:
        x["wiki_bio_test_idx"]  -> int
        x["gpt3_sentences"]     -> List[str]
        x["gpt3_text_samples"]  -> List[str]
    """

    def __init__(
        self,
        extractor: ClaimExtractor,
        use_pronoun_resolution: bool = True,
    ) -> None:

        self.extractor = extractor

        self.use_pronoun_resolution = use_pronoun_resolution

    # --------------------------------------------------

    def _prepare_example(
        self,
        original_sentences: List[str],
        sampled_passages: List[str],
    ) -> Tuple[Optional[str], List[str], List[List[str]]]:
        """
        Returns (primary_subject, resolved_original_sentences,
        sample_sentence_lists) - the latter one entry per sample
        passage, each entry a list of that sample's (possibly
        pronoun-resolved) sentences.
        """

        primary_subject = (
            identify_primary_subject(original_sentences[0])
            if original_sentences
            else None
        )

        if self.use_pronoun_resolution and primary_subject:
            resolved_original = resolve_pronouns(original_sentences, primary_subject)
        else:
            resolved_original = list(original_sentences)

        sample_sentence_lists: List[List[str]] = []

        for passage in sampled_passages:

            sentences = split_into_sentences(passage)

            if self.use_pronoun_resolution and primary_subject:
                sentences = resolve_pronouns(sentences, primary_subject)

            sample_sentence_lists.append(sentences)

        return primary_subject, resolved_original, sample_sentence_lists

    # --------------------------------------------------

    def _process_window(
        self,
        window: List[Tuple[int, Optional[str], List[str], List[List[str]]]],
    ) -> Dict[int, ExampleClaims]:
        """
        Extract one window of dataset examples as a single
        batched REBEL job, then reassemble per-document,
        per-sentence (originals) and per-sample-merged
        (samples) claim lists.
        """

        texts: List[str] = []

        original_spans: List[Tuple[int, int, int]] = []  # (dataset_id, start, end)

        sample_spans: List[Tuple[int, int, int, int]] = []  # (dataset_id, sample_index, start, end)

        subject_by_id: Dict[int, Optional[str]] = {}

        for dataset_id, primary_subject, original_sentences, sample_sentence_lists in window:

            subject_by_id[dataset_id] = primary_subject

            start = len(texts)
            texts.extend(original_sentences)
            original_spans.append((dataset_id, start, len(texts)))

            for sample_index, sentences in enumerate(sample_sentence_lists):
                start = len(texts)
                texts.extend(sentences)
                sample_spans.append((dataset_id, sample_index, start, len(texts)))

        claims_per_text = self.extractor.extract_batch(texts)

        results: Dict[int, ExampleClaims] = {}

        def entry_for(dataset_id: int) -> ExampleClaims:
            return results.setdefault(
                dataset_id,
                ExampleClaims(dataset_id=dataset_id, primary_subject=subject_by_id[dataset_id]),
            )

        for dataset_id, start, end in original_spans:
            entry_for(dataset_id).original_claims = claims_per_text[start:end]

        per_doc_samples: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)

        for dataset_id, sample_index, start, end in sample_spans:
            per_doc_samples[dataset_id].append((sample_index, start, end))

        for dataset_id, spans in per_doc_samples.items():

            spans.sort(key=lambda s: s[0])

            merged_per_sample: List[List[ExtractedClaim]] = []

            for _, start, end in spans:

                merged: List[ExtractedClaim] = []

                for claims in claims_per_text[start:end]:
                    merged.extend(claims)

                merged_per_sample.append(merged)

            entry_for(dataset_id).sampled_claims = merged_per_sample

        return results

    # --------------------------------------------------

    def build(
        self,
        dataset,
        output_path: str | Path,
        flush_every: int = 10,
        resume: bool = True,
        limit: int | None = None,
    ) -> ClaimCache:

        output_path = Path(output_path)

        cache = ClaimCache()

        if resume and output_path.exists():
            cache = ClaimCache.load(output_path)
            logger.info("Resuming: %s examples already cached.", len(cache))

        num_examples = len(dataset) if limit is None else min(limit, len(dataset))

        window: List[Tuple[int, Optional[str], List[str], List[List[str]]]] = []

        start_time = time.monotonic()

        texts_seen = 0

        for i in range(num_examples):

            x = dataset[i]

            dataset_id = x["wiki_bio_test_idx"]

            if cache.has(dataset_id):
                continue

            original_sentences = x["gpt3_sentences"]

            sampled_passages = x["gpt3_text_samples"]

            primary_subject, resolved_original, sample_sentence_lists = self._prepare_example(
                original_sentences, sampled_passages
            )

            texts_seen += len(resolved_original) + sum(len(s) for s in sample_sentence_lists)

            window.append((dataset_id, primary_subject, resolved_original, sample_sentence_lists))

            if len(window) >= flush_every:

                window_results = self._process_window(window)

                for dataset_id, entry in window_results.items():
                    cache.entries[dataset_id] = entry

                cache.save(output_path)

                elapsed = time.monotonic() - start_time

                rate = texts_seen / elapsed if elapsed > 0 else 0.0

                logger.info(
                    "Checkpointed %s examples (%.1f texts/sec so far).",
                    len(cache), rate,
                )

                window = []

        if window:

            window_results = self._process_window(window)

            for dataset_id, entry in window_results.items():
                cache.entries[dataset_id] = entry

            cache.save(output_path)

        elapsed = time.monotonic() - start_time

        rate = texts_seen / elapsed if elapsed > 0 else 0.0

        logger.info(
            "Claim cache build complete: %s examples in %.1fs (%.1f texts/sec, "
            "including model load time).",
            len(cache), elapsed, rate,
        )

        return cache


# =========================================================
# Canonical Belief Cache
# =========================================================
#
# Downstream of canonicalization (canonicalize_cache.py).
# Deliberately kept in this module (not canonicalize_cache.py)
# so every existing caller - evaluate_belief.py,
# evaluate_reliability.py, reliability/features.py,
# pipeline.py, sentence_runner.py - can keep importing
# ``BeliefCache`` from ``belief_stability.belief_cache``
# unchanged.


@dataclass
class ExampleBeliefs:

    dataset_id: int

    original_beliefs: List[List[Belief]] = field(default_factory=list)

    sampled_beliefs: List[List[Belief]] = field(default_factory=list)


class BeliefCache:
    """
    Read-only accessor over a built, canonicalized belief
    cache. Used online - no model is ever loaded by this
    class.
    """

    def __init__(self, entries: Dict[int, ExampleBeliefs] | None = None) -> None:

        self.entries: Dict[int, ExampleBeliefs] = entries or {}

    def has(self, dataset_id: int) -> bool:
        return dataset_id in self.entries

    def get(self, dataset_id: int) -> ExampleBeliefs:

        if dataset_id not in self.entries:
            raise KeyError(f"Dataset id {dataset_id} not found in belief cache.")

        return self.entries[dataset_id]

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, file_path: str | Path) -> None:

        file_path = Path(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: str | Path) -> "BeliefCache":

        file_path = Path(file_path)

        with open(file_path, "rb") as f:
            entries = pickle.load(f)

        logger.info("Loaded belief cache: %s examples from %s", len(entries), file_path)

        return cls(entries=entries)
