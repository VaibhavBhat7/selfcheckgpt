"""
---------------------------------------------------------
Reliability Aggregation Engine

File        : subgroups.py

Description
-----------
Consistent-hallucination subgroup: non-factual sentences
where SelfCheckGPT-NLI's own score falls below its ROC-
optimal (Youden's J) decision threshold, computed ONLY from
the TRAINING FOLD's rows (both classes), then applied as a
fixed cutoff to any other rows (test-fold or otherwise), so
it never leaks test-fold information.

This is an ABSOLUTE threshold, not a percentile of the non-
factual distribution's own shape. An earlier version of this
function used the median SelfCheckGPT score among non-factual
sentences - that guarantees exactly 50% of non-factual rows
fall "below the threshold" regardless of whether SelfCheckGPT
was actually confidently wrong on them (data-driven analysis
showed the median-based subgroup's mean member score was
~0.72, comfortably above any reasonable "SelfCheckGPT thinks
this is factual" cutoff). The ROC-optimal threshold is instead
the single decision boundary that best separates factual from
non-factual sentences on the training fold - the subgroup then
becomes "non-factual sentences SelfCheckGPT still misses even
under its own best possible operating point," which is what
this subgroup is meant to measure.
---------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.metrics import roc_curve

from .features import SentenceRow


def compute_subgroup_threshold(
    rows: Sequence[SentenceRow],
    train_ids: Sequence[int],
) -> float:
    """
    ROC-optimal (Youden's J = TPR - FPR, maximized) SelfCheckGPT
    score threshold, fit on rows (both label 0 and label 1)
    whose dataset_id is in ``train_ids``. Falls back to 0.5 if
    the training fold doesn't contain both classes (ROC is
    undefined with a single class).
    """

    train_id_set = set(train_ids)

    train_rows = [row for row in rows if row.dataset_id in train_id_set]

    labels = [row.label for row in train_rows]

    scores = [row.selfcheck_score for row in train_rows]

    if len(set(labels)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    # thresholds[0] is sklearn's synthetic "predict nothing positive"
    # point (max(score) + 1) - exclude it so the returned threshold
    # is always an actual, achievable cutoff.
    if len(thresholds) > 1:
        fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]

    youden_j = tpr - fpr

    best_idx = int(np.argmax(youden_j))

    return float(thresholds[best_idx])


def in_subgroup(row: SentenceRow, threshold: float) -> bool:

    return row.label == 1 and row.selfcheck_score < threshold


def filter_subgroup(
    rows: Sequence[SentenceRow],
    threshold: float,
) -> List[SentenceRow]:

    return [row for row in rows if in_subgroup(row, threshold)]
