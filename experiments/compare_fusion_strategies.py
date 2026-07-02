"""
Read-only comparison of alternative fusion architectures against the
current global logistic-regression fusion (reliability/aggregator.py),
motivated by diagnose_fusion.py's finding that the LR's globally-fit
intercept implies a decision boundary on selfcheck_score (~0.90) far
above the 0.5 convention used elsewhere, and that a naive 50/50 average
recovers most of the subgroup recall loss for a small aggregate cost.

Makes NO changes to reliability/ or belief_stability/ - every strategy
below is implemented locally in this script and evaluated on the exact
same nested document-level CV, ROC-optimal subgroup definition, and
pooled out-of-fold metrics as experiments/evaluate_reliability.py, so
numbers are directly comparable to that script's bs_graph row.

Strategies compared (all leak-safe: any fit parameter is estimated
from the outer fold's TRAINING documents only):

  1. lr_raw          - current production fusion, unchanged, as a
                        reference point.
  2. lr_standardized - same architecture, features z-scored (scaler
                        fit on train fold) before the logistic
                        regression. Isolates whether the scale/
                        intercept-calibration issue alone explains
                        the regression, with zero architecture change.
  3. fixed_avg        - pred = 0.5*selfcheck + 0.5*belief_risk, no
                        fitting. Reference floor for "does naive
                        averaging even work."
  4. tuned_avg        - pred = w*selfcheck + (1-w)*belief_risk, w
                        selected via inner CV (same non-inferiority-
                        constrained rule as alpha_selection.py),
                        baseline w=1.0 (ignore belief_stability
                        entirely).
  5. evidence_gated   - fall back fully to selfcheck when the belief
                        graph has no support/contradict evidence for
                        a sentence (pure ABSENT); otherwise 50/50
                        blend. Gates on SECONDARY signal's evidence.
  6. cascade          - trust selfcheck alone when its own score is
                        already above the fold's ROC-optimal
                        threshold (SelfCheckGPT already confident);
                        otherwise blend. Gates on PRIMARY signal's
                        own confidence.

Usage
-----
    python experiments/compare_fusion_strategies.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from belief_stability.belief_cache import BeliefCache
from belief_stability.config import BeliefStabilityConfig
from belief_stability.matcher.semantic import SemanticMatcher
from belief_stability.utils import setup_logger
from build_selfcheck_cache import load_selfcheck_cache

from reliability.aggregator import ReliabilityAggregator
from reliability.alpha_selection import ACTIVE_FEATURES_GRAPH, auc_pr, recall_at_threshold, select_alpha
from reliability.cv import document_level_kfold
from reliability.features import DocumentContext, SentenceRow, attach_graph_scores, build_dataset
from reliability.subgroups import compute_subgroup_threshold

logger = setup_logger(__name__)

WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

STRATEGIES = ["lr_raw", "lr_standardized", "fixed_avg", "tuned_avg", "evidence_gated", "cascade"]


def bs_risk(bs_score: float) -> float:
    return (1.0 - bs_score) / 2.0


def rows_to_matrix(rows, graph_map):

    return np.array(
        [
            [r.selfcheck_score, graph_map[(r.dataset_id, r.sentence_index)], r.support, r.absent, r.contradict]
            for r in rows
        ],
        dtype=float,
    )


def fit_lr_standardized(train_rows, graph_map_train):

    X_train = rows_to_matrix(train_rows, graph_map_train)
    y_train = [r.label for r in train_rows]

    scaler = StandardScaler().fit(X_train)

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model.fit(scaler.transform(X_train), y_train)

    return scaler, model


def predict_lr_standardized(scaler, model, rows, graph_map):

    X = rows_to_matrix(rows, graph_map)

    return model.predict_proba(scaler.transform(X))[:, 1]


def select_weight(
    train_rows,
    contexts,
    alpha_star: float,
    absent_discount: float,
    inner_k: int = 3,
    seed: int = 42,
):
    """
    Inner-CV weight selection for tuned_avg, mirroring
    alpha_selection.select_alpha's non-inferiority-constrained rule:
    w* = argmax_w Recall_sub(w) s.t. AUC_agg(w) >= AUC_agg(1.0) - epsilon,
    where w=1.0 means "ignore belief_stability entirely" (pure
    selfcheck baseline) and epsilon is 1 SE of the baseline's inner-
    fold AUC_agg.
    """

    train_doc_ids = sorted({r.dataset_id for r in train_rows})

    inner_folds = list(document_level_kfold(train_doc_ids, k=inner_k, seed=seed))

    per_w_agg = {w: [] for w in WEIGHT_GRID}
    per_w_sub = {w: [] for w in WEIGHT_GRID}

    for inner_train_ids, inner_test_ids in inner_folds:

        inner_train_set, inner_test_set = set(inner_train_ids), set(inner_test_ids)

        inner_train_rows = [r for r in train_rows if r.dataset_id in inner_train_set]
        inner_test_rows = [r for r in train_rows if r.dataset_id in inner_test_set]

        inner_graph = attach_graph_scores(inner_test_rows, contexts, alpha_star, absent_discount)

        threshold = compute_subgroup_threshold(inner_train_rows, inner_train_ids)

        labels = [r.label for r in inner_test_rows]

        for w in WEIGHT_GRID:

            preds = [
                w * r.selfcheck_score + (1 - w) * bs_risk(inner_graph[(r.dataset_id, r.sentence_index)])
                for r in inner_test_rows
            ]

            per_w_agg[w].append(auc_pr(labels, preds))

            sub_pairs = [
                (r.label, p) for r, p in zip(inner_test_rows, preds)
                if r.label == 1 and r.selfcheck_score < threshold
            ]

            per_w_sub[w].append(
                recall_at_threshold([l for l, _ in sub_pairs], [p for _, p in sub_pairs]) if sub_pairs else float("nan")
            )

    mean_agg = {w: mean(v) for w, v in per_w_agg.items()}
    mean_sub = {
        w: (mean(clean) if (clean := [v for v in vals if v == v]) else float("-inf"))
        for w, vals in per_w_sub.items()
    }

    baseline_vals = per_w_agg[1.0]
    baseline_mean = mean(baseline_vals)
    epsilon = pstdev(baseline_vals) / (len(baseline_vals) ** 0.5) if len(baseline_vals) > 1 else 0.0

    eligible = [w for w in WEIGHT_GRID if mean_agg[w] >= baseline_mean - epsilon]
    if not eligible:
        eligible = [1.0]

    w_star = max(eligible, key=lambda w: mean_sub[w])

    return w_star


def main() -> None:

    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]

    config = BeliefStabilityConfig.from_yaml(str(PROJECT_ROOT / "configs" / "default.yaml"))

    belief_cache = BeliefCache.load(str(PROJECT_ROOT / "experiments" / "results" / "belief_cache.pkl"))

    selfcheck_cache = load_selfcheck_cache(str(PROJECT_ROOT / "experiments" / "results" / "selfcheck_cache.pkl"))

    embeddings_path = PROJECT_ROOT / "experiments" / "results" / "belief_cache.pkl.embeddings.pkl"

    semantic_matcher = SemanticMatcher.load(embeddings_path) if embeddings_path.exists() else SemanticMatcher()

    rows, contexts = build_dataset(dataset, belief_cache, selfcheck_cache, config, semantic_matcher)

    doc_ids = sorted({r.dataset_id for r in rows})

    outer_folds = list(document_level_kfold(doc_ids, k=5, seed=42))

    oof_agg = {s: [] for s in STRATEGIES}
    oof_sub = {s: [] for s in STRATEGIES}

    fold_weight_stars = []

    for fold_index, (train_ids, test_ids) in enumerate(outer_folds):

        train_id_set, test_id_set = set(train_ids), set(test_ids)

        train_rows = [r for r in rows if r.dataset_id in train_id_set]
        test_rows = [r for r in rows if r.dataset_id in test_id_set]

        threshold = compute_subgroup_threshold(train_rows, train_ids)

        subgroup_mask = [r.label == 1 and r.selfcheck_score < threshold for r in test_rows]

        alpha_star, _ = select_alpha(
            train_rows, contexts, inner_k=3, seed=42, absent_discount=config.absent_discount,
        )

        train_graph = attach_graph_scores(train_rows, contexts, alpha_star, config.absent_discount)
        test_graph = attach_graph_scores(test_rows, contexts, alpha_star, config.absent_discount)

        labels = [r.label for r in test_rows]

        preds_by_strategy = {}

        # ---- 1. lr_raw (current production fusion) ----

        X_train = [
            {"selfcheckgpt": r.selfcheck_score, "belief_stability": train_graph[(r.dataset_id, r.sentence_index)],
             "support": r.support, "absent": r.absent, "contradict": r.contradict}
            for r in train_rows
        ]
        X_test = [
            {"selfcheckgpt": r.selfcheck_score, "belief_stability": test_graph[(r.dataset_id, r.sentence_index)],
             "support": r.support, "absent": r.absent, "contradict": r.contradict}
            for r in test_rows
        ]
        y_train = [r.label for r in train_rows]

        lr_model = ReliabilityAggregator(ACTIVE_FEATURES_GRAPH).fit(X_train, y_train)
        preds_by_strategy["lr_raw"] = lr_model.predict_proba(X_test).tolist()

        # ---- 2. lr_standardized ----

        scaler, std_model = fit_lr_standardized(train_rows, train_graph)
        preds_by_strategy["lr_standardized"] = predict_lr_standardized(scaler, std_model, test_rows, test_graph).tolist()

        # ---- 3. fixed_avg ----

        preds_by_strategy["fixed_avg"] = [
            0.5 * r.selfcheck_score + 0.5 * bs_risk(test_graph[(r.dataset_id, r.sentence_index)])
            for r in test_rows
        ]

        # ---- 4. tuned_avg ----

        w_star = select_weight(train_rows, contexts, alpha_star, config.absent_discount, inner_k=3, seed=42)
        fold_weight_stars.append(w_star)

        preds_by_strategy["tuned_avg"] = [
            w_star * r.selfcheck_score + (1 - w_star) * bs_risk(test_graph[(r.dataset_id, r.sentence_index)])
            for r in test_rows
        ]

        # ---- 5. evidence_gated ----

        preds_by_strategy["evidence_gated"] = [
            r.selfcheck_score if (r.support + r.contradict) == 0
            else 0.5 * r.selfcheck_score + 0.5 * bs_risk(test_graph[(r.dataset_id, r.sentence_index)])
            for r in test_rows
        ]

        # ---- 6. cascade (gate on selfcheck's own confidence vs train-fold threshold) ----

        preds_by_strategy["cascade"] = [
            r.selfcheck_score if r.selfcheck_score >= threshold
            else 0.5 * r.selfcheck_score + 0.5 * bs_risk(test_graph[(r.dataset_id, r.sentence_index)])
            for r in test_rows
        ]

        for strategy in STRATEGIES:

            preds = preds_by_strategy[strategy]

            oof_agg[strategy].extend(zip(labels, preds))

            sub_pairs = [(l, p) for l, p, m in zip(labels, preds, subgroup_mask) if m]
            oof_sub[strategy].extend(sub_pairs)

        logger.info(
            "Fold %d done (alpha*=%.2f, tuned_avg w*=%.2f, subgroup n=%d)",
            fold_index, alpha_star, w_star, sum(subgroup_mask),
        )

    # =================================================================
    # Report
    # =================================================================

    logger.info("=" * 78)
    logger.info("FUSION STRATEGY COMPARISON (pooled out-of-fold, 5 outer folds)")
    logger.info("=" * 78)

    logger.info("tuned_avg w* per fold: %s", ["%.2f" % w for w in fold_weight_stars])

    logger.info("%-18s %14s %20s", "strategy", "agg_auc_pr", "subgroup_recall@0.5")

    for strategy in STRATEGIES:

        agg_labels = [l for l, _ in oof_agg[strategy]]
        agg_preds = [p for _, p in oof_agg[strategy]]
        agg_score = auc_pr(agg_labels, agg_preds)

        sub_labels = [l for l, _ in oof_sub[strategy]]
        sub_preds = [p for _, p in oof_sub[strategy]]
        sub_score = recall_at_threshold(sub_labels, sub_preds) if sub_preds else float("nan")

        logger.info("%-18s %14.2f %20.2f", strategy, agg_score, sub_score)

    logger.info("Done.")


if __name__ == "__main__":
    main()
