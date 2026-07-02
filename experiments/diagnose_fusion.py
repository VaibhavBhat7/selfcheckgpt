"""
Read-only diagnostic: WHY does the fused Belief Stability arms
(bs_baseline / bs_bayesian / bs_graph) underperform plain SelfCheckGPT
on the ROC-optimal consistent-hallucination subgroup (see
reliability/subgroups.py)?

Makes NO changes to reliability/ or belief_stability/ - reuses the
exact same nested-CV structure, ReliabilityAggregator, and subgroup
definition as experiments/evaluate_reliability.py, purely to inspect:

  1. Learned logistic regression coefficients (raw + standardized).
  2. Correlation between SelfCheckGPT score and each Belief
     Stability feature (belief_stability score, support, absent,
     contradict), overall / non-factual-only / subgroup-only.
  3. Feature distributions for subgroup members the fused model
     gets right (hit) vs wrong (miss).
  4/6. Whether Belief Stability is contradicting the correct label
     on subgroup members, or whether the model is just weighting it
     incorrectly (printed as raw evidence, interpreted in the report).
  5. Two hand-computed alternative fusions (simple weighted average,
     evidence-gated blend) evaluated on the SAME subgroup, for
     comparison - not wired into any production code path.

Usage
-----
    python experiments/diagnose_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, median, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datasets import load_dataset
from scipy import stats as scipy_stats

from belief_stability.belief_cache import BeliefCache
from belief_stability.config import BeliefStabilityConfig
from belief_stability.matcher.semantic import SemanticMatcher
from belief_stability.utils import setup_logger
from build_selfcheck_cache import load_selfcheck_cache

from reliability.aggregator import ReliabilityAggregator
from reliability.alpha_selection import ACTIVE_FEATURES_GRAPH, auc_pr, recall_at_threshold, select_alpha
from reliability.cv import document_level_kfold
from reliability.features import attach_graph_scores, build_dataset
from reliability.subgroups import compute_subgroup_threshold

logger = setup_logger(__name__)

ARM_ATTR = {"bs_baseline": "baseline_score", "bs_bayesian": "bayesian_score", "bs_graph": None}


def pearson_spearman(x, y):

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")

    r_p = float(np.corrcoef(x, y)[0, 1])
    r_s = float(scipy_stats.spearmanr(x, y).correlation)

    return r_p, r_s


def bs_value_for(row, attr, graph_map):

    if attr is not None:
        return getattr(row, attr)

    return graph_map[(row.dataset_id, row.sentence_index)]


def main() -> None:

    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]

    config = BeliefStabilityConfig.from_yaml(str(PROJECT_ROOT / "configs" / "default.yaml"))

    belief_cache = BeliefCache.load(str(PROJECT_ROOT / "experiments" / "results" / "belief_cache.pkl"))

    selfcheck_cache = load_selfcheck_cache(str(PROJECT_ROOT / "experiments" / "results" / "selfcheck_cache.pkl"))

    embeddings_path = PROJECT_ROOT / "experiments" / "results" / "belief_cache.pkl.embeddings.pkl"

    semantic_matcher = SemanticMatcher.load(embeddings_path) if embeddings_path.exists() else SemanticMatcher()

    rows, contexts = build_dataset(dataset, belief_cache, selfcheck_cache, config, semantic_matcher)

    doc_ids = sorted({row.dataset_id for row in rows})

    outer_folds = list(document_level_kfold(doc_ids, k=5, seed=42))

    coef_records = {arm: [] for arm in ARM_ATTR}

    hit_rows, miss_rows = [], []

    subgroup_pool = []  # (arm, fold, row, bs_val, pred) for every subgroup test row, every arm

    alpha_stars = []

    graph_map_at_representative_alpha = None  # filled on last fold, alpha=0.3, for correlation section

    oof_full = {"label": [], "selfcheck": [], "lr_pred": [], "simple_avg": [], "gated": []}

    for fold_index, (train_ids, test_ids) in enumerate(outer_folds):

        train_id_set, test_id_set = set(train_ids), set(test_ids)

        train_rows = [r for r in rows if r.dataset_id in train_id_set]
        test_rows = [r for r in rows if r.dataset_id in test_id_set]

        threshold = compute_subgroup_threshold(train_rows, train_ids)

        subgroup_test_rows = [r for r in test_rows if r.label == 1 and r.selfcheck_score < threshold]

        alpha_star, _ = select_alpha(
            train_rows, contexts, inner_k=3, seed=42, absent_discount=config.absent_discount,
        )
        alpha_stars.append(alpha_star)

        train_graph = attach_graph_scores(train_rows, contexts, alpha_star, config.absent_discount)
        test_graph = attach_graph_scores(test_rows, contexts, alpha_star, config.absent_discount)

        for arm, attr in ARM_ATTR.items():

            train_map = train_graph if attr is None else None
            test_map = test_graph if attr is None else None

            X_train = [
                {
                    "selfcheckgpt": r.selfcheck_score,
                    "belief_stability": bs_value_for(r, attr, train_map),
                    "support": r.support, "absent": r.absent, "contradict": r.contradict,
                }
                for r in train_rows
            ]
            X_test = [
                {
                    "selfcheckgpt": r.selfcheck_score,
                    "belief_stability": bs_value_for(r, attr, test_map),
                    "support": r.support, "absent": r.absent, "contradict": r.contradict,
                }
                for r in test_rows
            ]

            y_train = [r.label for r in train_rows]

            model = ReliabilityAggregator(ACTIVE_FEATURES_GRAPH).fit(X_train, y_train)

            preds = model.predict_proba(X_test)

            raw_coef = dict(zip(ACTIVE_FEATURES_GRAPH, model.model.coef_[0].tolist()))
            raw_coef["intercept"] = float(model.model.intercept_[0])

            X_train_arr = np.array([[r[f] for f in ACTIVE_FEATURES_GRAPH] for r in X_train])
            feature_std = dict(zip(ACTIVE_FEATURES_GRAPH, X_train_arr.std(axis=0).tolist()))

            std_coef = {f: raw_coef[f] * feature_std[f] for f in ACTIVE_FEATURES_GRAPH}

            coef_records[arm].append({"raw": raw_coef, "standardized": std_coef, "feature_std": feature_std})

            test_lookup = {
                (r.dataset_id, r.sentence_index): (bs_value_for(r, attr, test_map), p)
                for r, p in zip(test_rows, preds)
            }

            for r in subgroup_test_rows:

                key = (r.dataset_id, r.sentence_index)

                if key not in test_lookup:
                    continue

                bs_val, pred = test_lookup[key]

                record = {
                    "arm": arm, "fold": fold_index,
                    "selfcheck_score": r.selfcheck_score, "belief_stability": bs_val,
                    "support": r.support, "absent": r.absent, "contradict": r.contradict,
                    "pred": pred,
                }

                subgroup_pool.append(record)

                if arm == "bs_graph":
                    (hit_rows if pred >= 0.5 else miss_rows).append(record)

            if arm == "bs_graph":

                def bs_risk(bs_score: float) -> float:
                    return (1.0 - bs_score) / 2.0

                for r, p in zip(test_rows, preds):

                    bs_val, _ = test_lookup[(r.dataset_id, r.sentence_index)]

                    evidence = r.support + r.contradict

                    simple_avg = 0.5 * r.selfcheck_score + 0.5 * bs_risk(bs_val)

                    gated = (
                        r.selfcheck_score if evidence == 0
                        else 0.5 * r.selfcheck_score + 0.5 * bs_risk(bs_val)
                    )

                    oof_full["label"].append(r.label)
                    oof_full["selfcheck"].append(r.selfcheck_score)
                    oof_full["lr_pred"].append(float(p))
                    oof_full["simple_avg"].append(simple_avg)
                    oof_full["gated"].append(gated)

        if fold_index == len(outer_folds) - 1:
            graph_map_at_representative_alpha = attach_graph_scores(rows, contexts, 0.3, config.absent_discount)

    # =================================================================
    # 1. Learned logistic regression coefficients
    # =================================================================

    logger.info("=" * 78)
    logger.info("1. LOGISTIC REGRESSION COEFFICIENTS (mean +/- std across 5 outer folds)")
    logger.info("=" * 78)

    for arm in ARM_ATTR:

        logger.info("-- %s --", arm)

        for kind in ["raw", "standardized"]:

            logger.info("  %s coefficients:", kind)

            for feat in ACTIVE_FEATURES_GRAPH:

                vals = [rec[kind][feat] for rec in coef_records[arm]]

                logger.info(
                    "    %-14s %+.4f +/- %.4f   (per-fold: %s)",
                    feat, mean(vals), pstdev(vals) if len(vals) > 1 else 0.0,
                    ["%+.3f" % v for v in vals],
                )

        intercepts = [rec["raw"]["intercept"] for rec in coef_records[arm]]

        logger.info("  intercept (raw): %+.4f +/- %.4f", mean(intercepts), pstdev(intercepts) if len(intercepts) > 1 else 0.0)

    logger.info("alpha* per fold: %s", ["%.2f" % a for a in alpha_stars])

    # =================================================================
    # 2. Correlation: SelfCheckGPT score vs each Belief Stability feature
    # =================================================================

    logger.info("=" * 78)
    logger.info("2. CORRELATIONS: selfcheck_score vs Belief Stability features")
    logger.info("=" * 78)

    selfcheck_all = [r.selfcheck_score for r in rows]

    nonfactual_rows = [r for r in rows if r.label == 1]

    subgroup_rows_dedup = {(r["fold"], r["arm"]): None for r in subgroup_pool}  # placeholder unused

    # subgroup membership is arm-independent (defined purely by selfcheck
    # score vs train-fold threshold), so pull one arm's worth of rows to
    # avoid triple-counting across arms.
    subgroup_only = [r for r in subgroup_pool if r["arm"] == "bs_graph"]

    def report_correlation(name, x_vals, y_vals):
        r_p, r_s = pearson_spearman(x_vals, y_vals)
        logger.info("  %-45s pearson=%+.3f  spearman=%+.3f  (n=%d)", name, r_p, r_s, len(x_vals))

    for feat_name, per_row_getter in [
        ("baseline_score", lambda r: r.baseline_score),
        ("bayesian_score", lambda r: r.bayesian_score),
        ("graph_score (alpha=0.3)", lambda r: graph_map_at_representative_alpha[(r.dataset_id, r.sentence_index)]),
        ("support", lambda r: r.support),
        ("absent", lambda r: r.absent),
        ("contradict", lambda r: r.contradict),
    ]:

        report_correlation(
            f"[all rows] selfcheck vs {feat_name}",
            [r.selfcheck_score for r in rows], [per_row_getter(r) for r in rows],
        )
        report_correlation(
            f"[non-factual only] selfcheck vs {feat_name}",
            [r.selfcheck_score for r in nonfactual_rows], [per_row_getter(r) for r in nonfactual_rows],
        )

    report_correlation(
        "[subgroup only] selfcheck vs belief_stability(graph)",
        [r["selfcheck_score"] for r in subgroup_only], [r["belief_stability"] for r in subgroup_only],
    )

    # =================================================================
    # 3. Feature distributions: hit vs miss (bs_graph arm, subgroup only)
    # =================================================================

    logger.info("=" * 78)
    logger.info("3. SUBGROUP FEATURE DISTRIBUTIONS: hit (pred>=0.5) vs miss (pred<0.5), bs_graph arm")
    logger.info("=" * 78)

    logger.info(
        "n_hit=%d (%.1f%%)  n_miss=%d (%.1f%%)  n_subgroup_total=%d",
        len(hit_rows), 100 * len(hit_rows) / max(1, len(hit_rows) + len(miss_rows)),
        len(miss_rows), 100 * len(miss_rows) / max(1, len(hit_rows) + len(miss_rows)),
        len(hit_rows) + len(miss_rows),
    )

    for feat in ["selfcheck_score", "belief_stability", "support", "absent", "contradict"]:

        hit_vals = [r[feat] for r in hit_rows]
        miss_vals = [r[feat] for r in miss_rows]

        if hit_vals and miss_vals:
            u_stat, p_val = scipy_stats.mannwhitneyu(hit_vals, miss_vals, alternative="two-sided")
        else:
            u_stat, p_val = float("nan"), float("nan")

        logger.info(
            "  %-16s hit: mean=%.4f median=%.4f   miss: mean=%.4f median=%.4f   MannWhitney p=%.4g",
            feat,
            mean(hit_vals) if hit_vals else float("nan"), median(hit_vals) if hit_vals else float("nan"),
            mean(miss_vals) if miss_vals else float("nan"), median(miss_vals) if miss_vals else float("nan"),
            p_val,
        )

    # =================================================================
    # 4/6. Is Belief Stability contradicting the correct label on
    # subgroup members, or is it just uninformative/outweighted?
    # =================================================================

    logger.info("=" * 78)
    logger.info("4. BELIEF STABILITY SIGNAL DIRECTION ON SUBGROUP MEMBERS (bs_graph)")
    logger.info("=" * 78)
    logger.info(
        "Recall: belief_stability score is POSITIVE when the belief looks "
        "SUPPORTED/factual, NEGATIVE when CONTRADICTED/non-factual - opposite "
        "polarity from selfcheck_score. Subgroup members are all TRUE "
        "non-factual sentences, so a CORRECT complementary signal here would "
        "be negative (agreeing with the true label, disagreeing with "
        "SelfCheckGPT's wrong call)."
    )

    bs_vals = [r["belief_stability"] for r in subgroup_only]

    n_positive = sum(1 for v in bs_vals if v > 0.05)
    n_negative = sum(1 for v in bs_vals if v < -0.05)
    n_near_zero = len(bs_vals) - n_positive - n_negative

    logger.info(
        "  Of %d subgroup members: belief_stability > 0.05 (echoes SelfCheckGPT's WRONG 'factual' call)=%d (%.1f%%), "
        "< -0.05 (CORRECTLY signals non-factual)=%d (%.1f%%), near-zero (no usable evidence)=%d (%.1f%%)",
        len(bs_vals), n_positive, 100 * n_positive / len(bs_vals),
        n_negative, 100 * n_negative / len(bs_vals),
        n_near_zero, 100 * n_near_zero / len(bs_vals),
    )

    evidence_present = [r for r in subgroup_only if (r["support"] + r["contradict"]) > 0]
    evidence_absent = [r for r in subgroup_only if (r["support"] + r["contradict"]) == 0]

    logger.info(
        "  Subgroup members with ANY support/contradict evidence (not pure ABSENT): %d (%.1f%%); "
        "pure ABSENT (no belief-graph evidence at all): %d (%.1f%%)",
        len(evidence_present), 100 * len(evidence_present) / len(bs_vals),
        len(evidence_absent), 100 * len(evidence_absent) / len(bs_vals),
    )

    # =================================================================
    # 4b. Is the "92.3% correctly negative on the subgroup" number real
    # signal, or a base-rate artifact of the Bayesian/graph scorer
    # treating ABSENCE as partial negative evidence (formula: beta
    # grows with absent_weight regardless of ground truth)? Test by
    # checking the SAME "score < -0.05" rate on FACTUAL rows, where a
    # truly complementary signal should rarely fire negative.
    # =================================================================

    logger.info("=" * 78)
    logger.info("4b. BASE-RATE CHECK: is 'belief_stability < -0.05' just an absence artifact?")
    logger.info("=" * 78)

    factual_rows = [r for r in rows if r.label == 0]

    def negative_rate(row_list, label_name):
        vals = [graph_map_at_representative_alpha[(r.dataset_id, r.sentence_index)] for r in row_list]
        n_neg = sum(1 for v in vals if v < -0.05)
        logger.info(
            "  %-30s n=%-5d belief_stability(graph) < -0.05: %d (%.1f%%)",
            label_name, len(row_list), n_neg, 100 * n_neg / len(row_list) if row_list else float("nan"),
        )
        return vals

    negative_rate(factual_rows, "factual rows (label=0)")
    negative_rate(nonfactual_rows, "non-factual rows (label=1)")

    factual_evidence = [r for r in factual_rows if (r.support + r.contradict) > 0]
    factual_absent_only = [r for r in factual_rows if (r.support + r.contradict) == 0]

    negative_rate(factual_evidence, "factual rows WITH support/contradict evidence")
    negative_rate(factual_absent_only, "factual rows PURE ABSENT (no evidence)")

    # =================================================================
    # 5. Alternative fusions on the SAME subgroup (diagnostic only)
    # =================================================================

    logger.info("=" * 78)
    logger.info("5. ALTERNATIVE FUSIONS ON THE SUBGROUP (bs_graph belief_stability signal, diagnostic only)")
    logger.info("=" * 78)

    def bs_risk(bs_score: float) -> float:
        # map belief_stability [-1, 1] (supported..contradicted) to a
        # "risk of non-factual" scale comparable to selfcheck_score [0, 1]
        return (1.0 - bs_score) / 2.0

    selfcheck_only_recall = recall_at_threshold(
        [1] * len(subgroup_only), [r["selfcheck_score"] for r in subgroup_only],
    )

    lr_fused_recall = recall_at_threshold(
        [1] * len(subgroup_only), [r["pred"] for r in subgroup_only],
    )

    simple_avg_preds = [0.5 * r["selfcheck_score"] + 0.5 * bs_risk(r["belief_stability"]) for r in subgroup_only]
    simple_avg_recall = recall_at_threshold([1] * len(subgroup_only), simple_avg_preds)

    gated_preds = []
    for r in subgroup_only:
        evidence = r["support"] + r["contradict"]
        if evidence == 0:
            gated_preds.append(r["selfcheck_score"])  # no belief-graph evidence -> fall back fully to SelfCheckGPT
        else:
            gated_preds.append(0.5 * r["selfcheck_score"] + 0.5 * bs_risk(r["belief_stability"]))
    gated_recall = recall_at_threshold([1] * len(subgroup_only), gated_preds)

    logger.info("  SelfCheckGPT alone (raw score >= 0.5):        recall=%.2f%%", selfcheck_only_recall)
    logger.info("  Logistic-regression fusion (bs_graph, as evaluated): recall=%.2f%%", lr_fused_recall)
    logger.info("  Simple 50/50 weighted average:                 recall=%.2f%%", simple_avg_recall)
    logger.info("  Evidence-gated blend (fallback to SelfCheckGPT when no support/contradict evidence): recall=%.2f%%", gated_recall)

    # =================================================================
    # 5b. Does the alternative fusions' subgroup recall gain come at
    # the cost of AGGREGATE precision? (pooled out-of-fold, whole
    # test population, not just the subgroup)
    # =================================================================

    logger.info("=" * 78)
    logger.info("5b. AGGREGATE (whole test population) AUC-PR, pooled out-of-fold")
    logger.info("=" * 78)

    labels_full = oof_full["label"]

    for name, key in [
        ("SelfCheckGPT alone", "selfcheck"),
        ("Logistic-regression fusion (bs_graph)", "lr_pred"),
        ("Simple 50/50 weighted average", "simple_avg"),
        ("Evidence-gated blend", "gated"),
    ]:
        agg = auc_pr(labels_full, oof_full[key])
        logger.info("  %-42s aggregate AUC-PR=%.2f", name, agg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
