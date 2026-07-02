"""
---------------------------------------------------------
Reliability Aggregation Engine

File        : alpha_selection.py

Description
-----------
Non-inferiority constrained selection of GraphScorer's
``alpha`` smoothing parameter, evaluated on the COMBINED
framework (SelfCheckGPT + Belief Stability, fused via
ReliabilityAggregator) - not on Belief Stability alone.

Selection rule
--------------
alpha* = argmax_alpha  Recall_sub(alpha)
          subject to    AUC_agg(alpha) >= AUC_agg(0) - epsilon

where epsilon is 1 standard error of AUC_agg(0) across the
inner CV folds (not a hand-picked constant), and AUC_agg(0)
is exactly the SelfCheckGPT+Bayesian fusion (alpha=0 makes
GraphScorer collapse to unpropagated Bayesian).

The subgroup metric is recall@0.5, not AUC-PR: the
consistent-hallucination subgroup (subgroups.py::in_subgroup)
is non-factual sentences ONLY (label == 1), i.e. a single-
class population by construction - AUC-PR is undefined
without both classes and silently returns NaN on it. Recall@
threshold (fraction of this hard population the fused
classifier correctly flags as non-factual) is well-defined
on a single-class population and directly answers the
question the subgroup exists to ask: "of the cases where
SelfCheckGPT is confidently wrong, how many does the fused
signal still catch?"

This is genuinely NESTED cross-validation: alpha is selected
using only the outer fold's TRAINING documents, split again
into inner folds - the outer TEST fold is never touched by
alpha selection.
---------------------------------------------------------
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

from sklearn.metrics import auc, precision_recall_curve

from .aggregator import ReliabilityAggregator
from .cv import document_level_kfold
from .features import DocumentContext, SentenceRow, attach_graph_scores
from .subgroups import compute_subgroup_threshold

ACTIVE_FEATURES_GRAPH = ["selfcheckgpt", "belief_stability", "support", "absent", "contradict"]

ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def auc_pr(labels: Sequence[int], scores: Sequence[float]) -> float:
    """
    AUC-PR (x100), NaN if a fold/subgroup has only one class
    present (precision_recall_curve is undefined otherwise).
    """

    if len(set(labels)) < 2:
        return float("nan")

    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)

    return float(auc(recall, precision) * 100)


def recall_at_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> float:
    """
    Recall (x100) among label==1 rows only: fraction with a
    predicted score >= ``threshold``. Well-defined even when
    ``labels`` is entirely positive (unlike AUC-PR) - this is
    what makes it usable on the consistent-hallucination
    subgroup, which is positive-only by construction.
    """

    positive_scores = [s for l, s in zip(labels, scores) if l == 1]

    if not positive_scores:
        return float("nan")

    hits = sum(1 for s in positive_scores if s >= threshold)

    return 100.0 * hits / len(positive_scores)


def _rows_to_signal_dicts(
    rows: Sequence[SentenceRow],
    graph_scores: Dict[Tuple[int, int], float],
) -> List[Dict[str, float]]:

    return [
        {
            "selfcheckgpt": row.selfcheck_score,
            "belief_stability": graph_scores[(row.dataset_id, row.sentence_index)],
            "support": row.support,
            "absent": row.absent,
            "contradict": row.contradict,
        }
        for row in rows
    ]


def _evaluate_alpha_on_fold(
    train_rows: List[SentenceRow],
    test_rows: List[SentenceRow],
    contexts: Dict[int, DocumentContext],
    alpha: float,
    absent_discount: float,
) -> Tuple[float, float]:

    train_graph = attach_graph_scores(train_rows, contexts, alpha, absent_discount)

    test_graph = attach_graph_scores(test_rows, contexts, alpha, absent_discount)

    X_train = _rows_to_signal_dicts(train_rows, train_graph)

    X_test = _rows_to_signal_dicts(test_rows, test_graph)

    y_train = [row.label for row in train_rows]

    y_test = [row.label for row in test_rows]

    model = ReliabilityAggregator(ACTIVE_FEATURES_GRAPH).fit(X_train, y_train)

    preds = model.predict_proba(X_test).tolist()

    agg_auc = auc_pr(y_test, preds)

    threshold = compute_subgroup_threshold(train_rows, [row.dataset_id for row in train_rows])

    sub_pairs = [
        (row.label, pred)
        for row, pred in zip(test_rows, preds)
        if row.label == 1 and row.selfcheck_score < threshold
    ]

    if sub_pairs:
        sub_labels, sub_preds = zip(*sub_pairs)
        sub_recall = recall_at_threshold(list(sub_labels), list(sub_preds))
    else:
        sub_recall = float("nan")

    return agg_auc, sub_recall


def select_alpha(
    train_rows: List[SentenceRow],
    contexts: Dict[int, DocumentContext],
    inner_k: int = 3,
    seed: int = 42,
    absent_discount: float = 0.5,
    alpha_grid: Sequence[float] = ALPHA_GRID,
) -> Tuple[float, Dict]:
    """
    Select alpha via inner CV on ``train_rows`` only (the
    outer fold's training documents). Returns (alpha*, diagnostics).
    """

    train_doc_ids = sorted({row.dataset_id for row in train_rows})

    inner_folds = list(document_level_kfold(train_doc_ids, k=inner_k, seed=seed))

    per_alpha_agg: Dict[float, List[float]] = {alpha: [] for alpha in alpha_grid}

    per_alpha_sub: Dict[float, List[float]] = {alpha: [] for alpha in alpha_grid}

    for inner_train_ids, inner_test_ids in inner_folds:

        inner_train_set = set(inner_train_ids)

        inner_test_set = set(inner_test_ids)

        inner_train_rows = [row for row in train_rows if row.dataset_id in inner_train_set]

        inner_test_rows = [row for row in train_rows if row.dataset_id in inner_test_set]

        for alpha in alpha_grid:

            agg_auc, sub_auc = _evaluate_alpha_on_fold(
                inner_train_rows, inner_test_rows, contexts, alpha, absent_discount,
            )

            per_alpha_agg[alpha].append(agg_auc)
            per_alpha_sub[alpha].append(sub_auc)

    mean_agg = {alpha: mean(values) for alpha, values in per_alpha_agg.items()}

    mean_sub = {
        alpha: (mean(clean) if (clean := [v for v in values if v == v]) else float("-inf"))
        for alpha, values in per_alpha_sub.items()
    }

    baseline_values = per_alpha_agg[0.0]

    baseline_mean = mean(baseline_values)

    baseline_se = (
        pstdev(baseline_values) / (len(baseline_values) ** 0.5)
        if len(baseline_values) > 1
        else 0.0
    )

    epsilon = baseline_se

    eligible = [alpha for alpha in alpha_grid if mean_agg[alpha] >= baseline_mean - epsilon]

    if not eligible:
        eligible = [0.0]

    alpha_star = max(eligible, key=lambda alpha: mean_sub[alpha])

    diagnostics = {
        "mean_agg": mean_agg,
        "mean_sub": mean_sub,
        "epsilon": epsilon,
        "baseline_mean_agg": baseline_mean,
        "eligible_alphas": eligible,
    }

    return alpha_star, diagnostics
