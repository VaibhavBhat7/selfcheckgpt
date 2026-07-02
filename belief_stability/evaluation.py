"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Evaluation

File        : evaluation.py

Description
-----------
AUC-PR evaluation helpers against the wiki_bio_gpt3_hallucination
human annotations. Factored out of
demo/experiments/selfcheckgpt-nli-on-wikibio.ipynb (cells
17-23), which previously defined these inline and only ever
used them for SelfCheckGPT-NLI's scores - never for Belief
Stability's. Both the notebook and experiments/evaluate_belief.py
now share this one implementation.

Three standard experiments (matching the existing notebook):
  - detect_false   : "is this sentence non-factual?"
  - detect_false_h : same, but only over examples that are
                      not entirely non-factual (harder/less
                      trivial - "detect False*" in the notebook)
  - detect_true     : "is this sentence factual?"
---------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

LABEL_MAPPING = {
    "accurate": 0.0,
    "minor_inaccurate": 0.5,
    "major_inaccurate": 1.0,
}


def build_human_labels(
    dataset,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Build the three human-label dictionaries used by the
    standard WikiBio AUC-PR experiments, keyed by
    ``wiki_bio_test_idx``.

    Returns
    -------
    (detect_false, detect_true, detect_false_hard)
    """

    detect_false: Dict[int, List[int]] = {}
    detect_true: Dict[int, List[int]] = {}
    detect_false_hard: Dict[int, List[int]] = {}

    for i in range(len(dataset)):

        x = dataset[i]

        idx = x["wiki_bio_test_idx"]

        raw_label = np.array(
            [LABEL_MAPPING[label] for label in x["annotation"]]
        )

        detect_false[idx] = (raw_label > 0.499).astype(np.int32).tolist()
        detect_true[idx] = (raw_label < 0.499).astype(np.int32).tolist()

        average_score = np.mean(raw_label)

        if average_score < 0.99:
            detect_false_hard[idx] = (raw_label > 0.99).astype(np.int32).tolist()

    return detect_false, detect_true, detect_false_hard


def unroll_pred(
    scores: Dict[int, List[float]],
    indices: List[int],
) -> List[float]:

    unrolled: List[float] = []

    for idx in indices:
        unrolled.extend(scores[idx])

    return unrolled


def get_pr_with_human_labels(
    preds: Dict[int, List[float]],
    human_labels: Dict[int, List[int]],
    pos_label: int = 1,
    oneminus_pred: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    indices = list(human_labels.keys())

    unrolled_preds = unroll_pred(preds, indices)

    if oneminus_pred:
        unrolled_preds = [1.0 - x for x in unrolled_preds]

    unrolled_labels = unroll_pred(human_labels, indices)

    assert len(unrolled_preds) == len(unrolled_labels)

    precision, recall, _ = precision_recall_curve(
        unrolled_labels, unrolled_preds, pos_label=pos_label
    )

    return precision, recall


def compute_auc(precision: np.ndarray, recall: np.ndarray) -> float:

    return float(auc(recall, precision) * 100)


def random_baseline(human_labels: Dict[int, List[int]]) -> float:

    values: List[int] = []

    for v in human_labels.values():
        values.extend(v)

    return float(np.mean(values))
