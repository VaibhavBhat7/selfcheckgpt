"""
---------------------------------------------------------
Reliability Aggregation Engine

File        : cv.py

Description
-----------
Document-level cross-validation splitting. Sentences from
the same WikiBio passage must never be split across
train/test - splitting must happen on wiki_bio_test_idx
values, not on the unrolled sentence rows, or correlated
sentences leak between train and test.

Used for both the outer loop (final reporting) and the
inner loop (alpha selection).
---------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

from sklearn.model_selection import KFold


def document_level_kfold(
    dataset_ids: Sequence[int],
    k: int,
    seed: int = 42,
) -> Iterator[Tuple[List[int], List[int]]]:
    """
    Yield (train_ids, test_ids) tuples, splitting on
    document ids (not sentence rows).
    """

    ids = list(dataset_ids)

    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    for train_idx, test_idx in kfold.split(ids):

        train_ids = [ids[i] for i in train_idx]
        test_ids = [ids[i] for i in test_idx]

        yield train_ids, test_ids
