"""
Reliability Aggregation evaluation: SelfCheckGPT alone vs
SelfCheckGPT + Belief Stability (baseline / bayesian / graph),
fused via logistic regression, on nested document-level CV.

Requires two offline caches to already exist:
    experiments/results/belief_cache.pkl        (build_belief_cache.py)
    experiments/results/selfcheck_cache.pkl      (build_selfcheck_cache.py)

This script never runs REBEL or SelfCheckNLI - it only reads the
caches, fits small logistic regressions, and evaluates. Should
complete in well under a minute.

Usage
-----
    python experiments/evaluate_reliability.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from belief_stability.belief_cache import BeliefCache
from belief_stability.config import BeliefStabilityConfig
from belief_stability.matcher.semantic import SemanticMatcher
from belief_stability.utils import setup_logger
from build_selfcheck_cache import load_selfcheck_cache

from reliability.aggregator import ReliabilityAggregator
from reliability.alpha_selection import ACTIVE_FEATURES_GRAPH, auc_pr, recall_at_threshold, select_alpha
from reliability.cascade import BeliefStabilityStage, ReliabilityCascade, SelfCheckGPTStage
from reliability.cv import document_level_kfold
from reliability.features import attach_graph_scores, build_dataset
from reliability.subgroups import compute_subgroup_threshold

logger = setup_logger(__name__)

ARMS = ["selfcheckgpt", "bs_baseline", "bs_bayesian", "bs_graph", "cascade"]


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset", default="potsawee/wiki_bio_gpt3_hallucination")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument(
        "--belief-cache",
        default=str(PROJECT_ROOT / "experiments" / "results" / "belief_cache.pkl"),
    )
    parser.add_argument(
        "--selfcheck-cache",
        default=str(PROJECT_ROOT / "experiments" / "results" / "selfcheck_cache.pkl"),
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "experiments" / "results")
    )
    parser.add_argument("--outer-k", type=int, default=5)
    parser.add_argument("--inner-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-nli-arbitration", action="store_true",
        help="Override config.use_nli_arbitration=True for this run (isolated experiment flag, "
             "does not touch configs/default.yaml).",
    )

    return parser.parse_args()


def _fit_and_eval_fixed_arm(
    train_rows, test_rows, belief_score_attr: str, absent_discount: float
):

    active_features = ["selfcheckgpt", "belief_stability", "support", "absent", "contradict"]

    X_train = [
        {
            "selfcheckgpt": row.selfcheck_score,
            "belief_stability": getattr(row, belief_score_attr),
            "support": row.support,
            "absent": row.absent,
            "contradict": row.contradict,
        }
        for row in train_rows
    ]

    X_test = [
        {
            "selfcheckgpt": row.selfcheck_score,
            "belief_stability": getattr(row, belief_score_attr),
            "support": row.support,
            "absent": row.absent,
            "contradict": row.contradict,
        }
        for row in test_rows
    ]

    y_train = [row.label for row in train_rows]
    y_test = [row.label for row in test_rows]

    model = ReliabilityAggregator(active_features).fit(X_train, y_train)

    preds = model.predict_proba(X_test).tolist()

    return preds, y_test


def main() -> None:

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.monotonic()

    # ------------------------------------------------------------
    # Load caches
    # ------------------------------------------------------------

    for path, label in [(args.belief_cache, "belief cache"), (args.selfcheck_cache, "selfcheck cache")]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No {label} found at {path}. Build it first "
                f"(experiments/build_belief_cache.py / experiments/build_selfcheck_cache.py)."
            )

    logger.info("Loading dataset %s [%s]...", args.dataset, args.split)
    dataset = load_dataset(args.dataset)[args.split]

    config = BeliefStabilityConfig.from_yaml(args.config)

    if args.use_nli_arbitration:
        config.use_nli_arbitration = True

    belief_cache = BeliefCache.load(args.belief_cache)

    selfcheck_cache = load_selfcheck_cache(args.selfcheck_cache)

    embeddings_path = Path(args.belief_cache).with_suffix(
        Path(args.belief_cache).suffix + ".embeddings.pkl"
    )
    semantic_matcher = (
        SemanticMatcher.load(embeddings_path) if embeddings_path.exists() else SemanticMatcher()
    )

    nli_arbitrator = None

    if config.use_nli_arbitration:
        from belief_stability.matcher.nli_arbitrator import NLIArbitrator
        logger.info("Loading NLIArbitrator (DeBERTa-v3-large-MNLI) for Tier 3 matching...")
        nli_arbitrator = NLIArbitrator()

    # ------------------------------------------------------------
    # Build feature table
    # ------------------------------------------------------------

    feature_start = time.monotonic()

    rows, contexts = build_dataset(
        dataset, belief_cache, selfcheck_cache, config, semantic_matcher, nli_arbitrator,
    )

    feature_elapsed = time.monotonic() - feature_start

    doc_ids = sorted({row.dataset_id for row in rows})

    logger.info(
        "Feature table built: %s sentence rows across %s documents in %.2fs.",
        len(rows), len(doc_ids), feature_elapsed,
    )

    if len(doc_ids) < args.outer_k:
        raise ValueError(
            f"Only {len(doc_ids)} documents available (need >= --outer-k={args.outer_k}). "
            "Build the caches for more examples, or lower --outer-k."
        )

    # ------------------------------------------------------------
    # Outer CV
    # ------------------------------------------------------------

    cv_start = time.monotonic()

    outer_folds = list(document_level_kfold(doc_ids, k=args.outer_k, seed=args.seed))

    fold_results = []  # rows for reliability_cv_results.csv

    alpha_rows = []  # rows for alpha_selection.csv

    oof_agg = {arm: [] for arm in ARMS}  # pooled (label, pred) across folds
    oof_sub = {arm: [] for arm in ARMS}

    for fold_index, (train_ids, test_ids) in enumerate(outer_folds):

        train_id_set, test_id_set = set(train_ids), set(test_ids)

        train_rows = [row for row in rows if row.dataset_id in train_id_set]
        test_rows = [row for row in rows if row.dataset_id in test_id_set]

        threshold = compute_subgroup_threshold(train_rows, train_ids)

        subgroup_mask = [
            row.label == 1 and row.selfcheck_score < threshold for row in test_rows
        ]

        # ---- Arm 1: SelfCheckGPT alone (no fitting) ----

        preds = [row.selfcheck_score for row in test_rows]
        labels = [row.label for row in test_rows]

        _record_arm(
            "selfcheckgpt", fold_index, preds, labels, subgroup_mask,
            fold_results, oof_agg, oof_sub,
        )

        # ---- Arms 2-3: SelfCheckGPT + Baseline / Bayesian ----

        for arm, attr in [("bs_baseline", "baseline_score"), ("bs_bayesian", "bayesian_score")]:

            preds, labels = _fit_and_eval_fixed_arm(
                train_rows, test_rows, attr, config.absent_discount
            )

            _record_arm(
                arm, fold_index, preds, labels, subgroup_mask,
                fold_results, oof_agg, oof_sub,
            )

        # ---- Arm 4: SelfCheckGPT + Graph (alpha selected via nested inner CV) ----

        alpha_star, diagnostics = select_alpha(
            train_rows, contexts, inner_k=args.inner_k, seed=args.seed,
            absent_discount=config.absent_discount,
        )

        train_graph = attach_graph_scores(train_rows, contexts, alpha_star, config.absent_discount)
        test_graph = attach_graph_scores(test_rows, contexts, alpha_star, config.absent_discount)

        X_train = [
            {
                "selfcheckgpt": row.selfcheck_score,
                "belief_stability": train_graph[(row.dataset_id, row.sentence_index)],
                "support": row.support, "absent": row.absent, "contradict": row.contradict,
            }
            for row in train_rows
        ]
        X_test = [
            {
                "selfcheckgpt": row.selfcheck_score,
                "belief_stability": test_graph[(row.dataset_id, row.sentence_index)],
                "support": row.support, "absent": row.absent, "contradict": row.contradict,
            }
            for row in test_rows
        ]

        model = ReliabilityAggregator(ACTIVE_FEATURES_GRAPH).fit(
            X_train, [row.label for row in train_rows]
        )
        preds = model.predict_proba(X_test).tolist()
        labels = [row.label for row in test_rows]

        _record_arm(
            "bs_graph", fold_index, preds, labels, subgroup_mask,
            fold_results, oof_agg, oof_sub,
        )

        alpha_rows.append({
            "fold": fold_index,
            "alpha_star": alpha_star,
            "epsilon": diagnostics["epsilon"],
            "baseline_mean_agg": diagnostics["baseline_mean_agg"],
            "eligible_alphas": ";".join(str(a) for a in diagnostics["eligible_alphas"]),
        })

        logger.info(
            "Fold %s: alpha*=%.2f (epsilon=%.3f, eligible=%s)",
            fold_index, alpha_star, diagnostics["epsilon"], diagnostics["eligible_alphas"],
        )

        # ---- Arm 5: Reliability Cascade (production default, see reliability/cascade.py) ----
        #
        # Reuses the SAME train-fold ROC-optimal threshold as the
        # subgroup definition (stage 1 gate) and the SAME alpha* as
        # the bs_graph arm above (stage 2's graph score), so this is
        # an apples-to-apples comparison against bs_graph - the only
        # difference is the fusion architecture itself.

        cascade = ReliabilityCascade([
            SelfCheckGPTStage(threshold=threshold),
            BeliefStabilityStage(
                contexts=contexts, alpha=alpha_star, absent_discount=config.absent_discount,
                blend_weight=config.cascade_blend_weight, is_terminal=True,
            ),
        ])

        cascade_scores = cascade.predict(test_rows)

        preds = [cascade_scores[(row.dataset_id, row.sentence_index)] for row in test_rows]
        labels = [row.label for row in test_rows]

        _record_arm(
            "cascade", fold_index, preds, labels, subgroup_mask,
            fold_results, oof_agg, oof_sub,
        )

    cv_elapsed = time.monotonic() - cv_start

    # ------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------

    _write_cv_results_csv(fold_results, output_dir / "reliability_cv_results.csv")

    summary_rows = _write_summary_csv(fold_results, output_dir / "reliability_summary.csv")

    _write_alpha_csv(alpha_rows, output_dir / "alpha_selection.csv")

    _plot_alpha_distribution(alpha_rows, output_dir / "alpha_distribution.png")

    _plot_pooled_pr_curves(
        oof_agg, output_dir / "pr_curve_reliability_aggregate.png", "Aggregate: Detect Non-Factual"
    )
    _plot_subgroup_recall_bar(
        oof_sub, output_dir / "subgroup_recall.png",
        "Consistent-Hallucination Subgroup (ROC-optimal): Recall@0.5",
    )

    headline_oof = {arm: oof_agg[arm] for arm in ["selfcheckgpt", "bs_graph", "cascade"]}
    _plot_pooled_pr_curves(
        headline_oof, output_dir / "pr_curve_headline_comparison.png",
        "SelfCheckGPT vs Logistic Regression Fusion vs Cascade",
    )

    # ------------------------------------------------------------
    # Paired significance tests across the 5 outer folds
    # ------------------------------------------------------------

    significance_rows = []

    for arm_a, arm_b, metric_key, metric_name in [
        ("cascade", "selfcheckgpt", "agg_auc_pr", "aggregate AUC-PR"),
        ("cascade", "selfcheckgpt", "sub_recall", "subgroup recall@0.5"),
        ("cascade", "bs_graph", "agg_auc_pr", "aggregate AUC-PR"),
        ("cascade", "bs_graph", "sub_recall", "subgroup recall@0.5"),
    ]:
        result = _paired_fold_test(fold_results, arm_a, arm_b, metric_key)
        result.update({"arm_a": arm_a, "arm_b": arm_b, "metric": metric_name})
        significance_rows.append(result)

        logger.info(
            "Paired test [%s] %s vs %s: mean_diff=%+.2f (95%% CI [%+.2f, %+.2f]), "
            "t=%.3f, p=%.4f, n_folds=%d -> %s",
            metric_name, arm_a, arm_b, result["mean_diff"], result["ci95_low"], result["ci95_high"],
            result["t_stat"], result["p_value"], result["n_folds"],
            "SIGNIFICANT (p<0.05)" if result["p_value"] < 0.05 else "not significant (p>=0.05)",
        )

    _write_significance_csv(significance_rows, output_dir / "significance_tests.csv")

    overall_elapsed = time.monotonic() - overall_start

    runtime_info = {
        "feature_build_seconds": round(feature_elapsed, 2),
        "cv_seconds": round(cv_elapsed, 2),
        "total_seconds": round(overall_elapsed, 2),
        "num_documents": len(doc_ids),
        "num_sentence_rows": len(rows),
        "outer_k": args.outer_k,
        "inner_k": args.inner_k,
        "belief_cache_bytes": Path(args.belief_cache).stat().st_size,
        "selfcheck_cache_bytes": Path(args.selfcheck_cache).stat().st_size,
        "note": (
            "belief_cache/selfcheck_cache build durations are logged by "
            "build_belief_cache.py / build_selfcheck_cache.py themselves at "
            "build time, not reproduced here - this file only times this script."
        ),
    }

    with open(output_dir / "reliability_runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime_info, f, indent=2)

    logger.info("Done in %.2fs. Summary:", overall_elapsed)

    for row in summary_rows:
        logger.info(
            "%-14s agg=%.2f±%.2f  subgroup_recall=%.2f±%.2f",
            row["arm"], row["agg_mean"], row["agg_std"], row["sub_recall_mean"], row["sub_recall_std"],
        )


def _record_arm(arm, fold_index, preds, labels, subgroup_mask, fold_results, oof_agg, oof_sub):

    agg_score = auc_pr(labels, preds)

    sub_pairs = [(l, p) for l, p, m in zip(labels, preds, subgroup_mask) if m]

    # The subgroup is non-factual sentences only (label == 1 by
    # construction, see subgroups.py::in_subgroup) - AUC-PR is
    # undefined without both classes. Recall@0.5 (of this hard,
    # confidently-wrong-per-SelfCheckGPT population, how much does
    # the fused signal still catch) is well-defined here instead.
    sub_score = (
        recall_at_threshold([l for l, _ in sub_pairs], [p for _, p in sub_pairs])
        if sub_pairs else float("nan")
    )

    fold_results.append({
        "fold": fold_index, "arm": arm, "agg_auc_pr": agg_score, "sub_recall": sub_score,
    })

    oof_agg[arm].extend(zip(labels, preds))
    oof_sub[arm].extend(sub_pairs)


def _write_cv_results_csv(fold_results, path):

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "arm", "agg_auc_pr", "sub_recall"])
        writer.writeheader()
        writer.writerows(fold_results)

    logger.info("Wrote %s", path)


def _write_summary_csv(fold_results, path):

    by_arm = {arm: {"agg": [], "sub": []} for arm in ARMS}

    for row in fold_results:
        by_arm[row["arm"]]["agg"].append(row["agg_auc_pr"])
        if row["sub_recall"] == row["sub_recall"]:  # skip NaN
            by_arm[row["arm"]]["sub"].append(row["sub_recall"])

    summary_rows = []

    for arm in ARMS:
        agg_vals = by_arm[arm]["agg"]
        sub_vals = by_arm[arm]["sub"]
        summary_rows.append({
            "arm": arm,
            "agg_mean": round(mean(agg_vals), 2),
            "agg_std": round(pstdev(agg_vals), 2) if len(agg_vals) > 1 else 0.0,
            "sub_recall_mean": round(mean(sub_vals), 2) if sub_vals else float("nan"),
            "sub_recall_std": round(pstdev(sub_vals), 2) if len(sub_vals) > 1 else 0.0,
            "n_folds": len(agg_vals),
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["arm", "agg_mean", "agg_std", "sub_recall_mean", "sub_recall_std", "n_folds"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Wrote %s", path)

    return summary_rows


def _paired_fold_test(fold_results, arm_a, arm_b, metric_key):
    """
    Paired t-test (scipy.stats.ttest_rel) on per-outer-fold metric
    values between two arms - each fold contributes one paired
    observation (arm_a's value, arm_b's value) since both arms are
    evaluated on the SAME outer test fold. With only outer_k folds
    (5 by default) this test has low power - reported n_folds and the
    95% CI on the mean difference alongside the p-value so a small,
    non-significant p-value isn't over-interpreted as "no effect",
    and a small p>=0.05 difference isn't overstated as "significant"
    either.
    """

    from scipy import stats as scipy_stats

    by_fold_a = {row["fold"]: row[metric_key] for row in fold_results if row["arm"] == arm_a}
    by_fold_b = {row["fold"]: row[metric_key] for row in fold_results if row["arm"] == arm_b}

    common_folds = sorted(set(by_fold_a) & set(by_fold_b))

    vals_a = [by_fold_a[f] for f in common_folds if by_fold_a[f] == by_fold_a[f] and by_fold_b[f] == by_fold_b[f]]
    vals_b = [by_fold_b[f] for f in common_folds if by_fold_a[f] == by_fold_a[f] and by_fold_b[f] == by_fold_b[f]]

    diffs = [a - b for a, b in zip(vals_a, vals_b)]

    n = len(diffs)

    mean_diff = mean(diffs) if diffs else float("nan")

    if n > 1:
        t_stat, p_value = scipy_stats.ttest_rel(vals_a, vals_b)
        se = pstdev(diffs) / (n ** 0.5)
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        ci_low, ci_high = mean_diff - t_crit * se, mean_diff + t_crit * se
    else:
        t_stat, p_value, ci_low, ci_high = float("nan"), float("nan"), float("nan"), float("nan")

    return {
        "mean_diff": mean_diff, "t_stat": float(t_stat), "p_value": float(p_value),
        "ci95_low": ci_low, "ci95_high": ci_high, "n_folds": n,
    }


def _write_significance_csv(rows, path):

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["arm_a", "arm_b", "metric", "mean_diff", "ci95_low", "ci95_high", "t_stat", "p_value", "n_folds"]
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %s", path)


def _write_alpha_csv(alpha_rows, path):

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["fold", "alpha_star", "epsilon", "baseline_mean_agg", "eligible_alphas"]
        )
        writer.writeheader()
        writer.writerows(alpha_rows)

    logger.info("Wrote %s", path)


def _plot_alpha_distribution(alpha_rows, path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = [row["fold"] for row in alpha_rows]
    alphas = [row["alpha_star"] for row in alpha_rows]

    plt.figure(figsize=(5.5, 4.0))
    plt.bar([str(f) for f in folds], alphas)
    plt.ylim(0, 1.0)
    plt.xlabel("Outer CV fold")
    plt.ylabel("Selected alpha*")
    plt.title("Selected Graph alpha per outer fold")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    logger.info("Wrote %s", path)


def _plot_subgroup_recall_bar(oof_sub_by_arm, path, title):
    """
    The subgroup (subgroups.py::in_subgroup) is non-factual
    sentences only - a single-class population by construction -
    so a precision-recall curve is undefined for it (unlike the
    aggregate plot). Recall@0.5, pooled across all outer folds,
    is plotted as a bar per arm instead.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(oof_sub_by_arm.keys())

    recalls = []

    for arm in arms:

        pairs = oof_sub_by_arm[arm]

        labels = [l for l, _ in pairs]
        preds = [p for _, p in pairs]

        recalls.append(recall_at_threshold(labels, preds) if pairs else 0.0)

    plt.figure(figsize=(5.5, 4.0))
    plt.bar(arms, recalls)
    plt.ylim(0, 100)
    plt.ylabel("Recall@0.5 (%)")
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    logger.info("Wrote %s", path)


def _plot_pooled_pr_curves(oof_by_arm, path, title):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(6.0, 4.8))

    for arm, pairs in oof_by_arm.items():

        if not pairs:
            continue

        labels = [l for l, _ in pairs]
        preds = [p for _, p in pairs]

        if len(set(labels)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(labels, preds, pos_label=1)

        plt.plot(recall, precision, label=arm)

    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
