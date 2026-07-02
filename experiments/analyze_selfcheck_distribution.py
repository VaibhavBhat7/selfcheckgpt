"""
Data-driven analysis of SelfCheckGPT-NLI's score distribution on the
full wiki_bio_gpt3_hallucination evaluation split, run BEFORE any
decision to redefine the consistent-hallucination subgroup
(reliability/subgroups.py). Answers, with evidence:

  1. What does the SelfCheckGPT score distribution look like overall,
     for factual sentences, and for non-factual sentences (further
     split by accurate / minor_inaccurate / major_inaccurate)?
  2. Is 0.5 a statistically meaningful decision threshold on this
     dataset, or is it just a default that happens to sit in a region
     where almost all non-factual sentences already score above it?
  3. What threshold would the data itself suggest (ROC-optimal /
     Youden's J, and a factual-distribution-calibrated threshold)?
  4. What subgroup size and composition would each of 4 candidate
     "consistent hallucination" subgroup definitions produce?

This script makes NO changes to reliability/subgroups.py or any other
evaluation code - it is read-only analysis over the two existing
offline caches (selfcheck_cache.pkl) plus the dataset's ground-truth
annotations.

Usage
-----
    python experiments/analyze_selfcheck_distribution.py
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
from sklearn.metrics import roc_curve

from belief_stability.evaluation import LABEL_MAPPING
from belief_stability.utils import setup_logger
from build_selfcheck_cache import load_selfcheck_cache

logger = setup_logger(__name__)

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


def collect_scores(dataset, selfcheck_cache):
    """
    Returns two parallel lists (scores, categories) over every
    sentence in the dataset that has both a cached SelfCheckGPT
    score and a ground-truth annotation. categories are the raw
    annotation strings: "accurate" / "minor_inaccurate" / "major_inaccurate".
    """

    scores: list[float] = []
    categories: list[str] = []

    for i in range(len(dataset)):

        x = dataset[i]

        idx = x["wiki_bio_test_idx"]

        if idx not in selfcheck_cache:
            continue

        sc_scores = selfcheck_cache[idx]

        annotations = x["annotation"]

        num_sentences = min(len(sc_scores), len(annotations))

        for s in range(num_sentences):
            scores.append(float(sc_scores[s]))
            categories.append(annotations[s])

    return scores, categories


def describe(name: str, values: list[float]) -> dict:

    arr = np.array(values, dtype=float)

    stats = {
        "name": name,
        "n": len(arr),
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "std": float(np.std(arr)) if len(arr) else float("nan"),
    }

    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(arr, p)) if len(arr) else float("nan")

    return stats


def print_stats(stats: dict) -> None:

    logger.info(
        "%-18s n=%-5d mean=%.4f median=%.4f std=%.4f",
        stats["name"], stats["n"], stats["mean"], stats["median"], stats["std"],
    )

    pct_str = "  ".join(f"p{p}={stats[f'p{p}']:.4f}" for p in PERCENTILES)

    logger.info("%-18s %s", "", pct_str)


def bimodality_check(values: list[float]) -> dict:
    """
    1-component vs 2-component GaussianMixture BIC comparison.
    Lower BIC wins; a 2-component model winning by a meaningful
    margin (and with reasonably separated, non-degenerate means)
    is evidence of bimodality. This is a standard, explainable,
    data-driven test - no ad hoc visual judgment call.
    """

    from sklearn.mixture import GaussianMixture

    arr = np.array(values, dtype=float).reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(arr)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(arr)

    bic1 = gmm1.bic(arr)
    bic2 = gmm2.bic(arr)

    means2 = sorted(gmm2.means_.ravel().tolist())

    return {
        "bic_1_component": float(bic1),
        "bic_2_component": float(bic2),
        "two_component_wins": bic2 < bic1,
        "two_component_means": means2,
        "two_component_weights": sorted(gmm2.weights_.ravel().tolist()),
    }


def roc_optimal_threshold(nonfactual_scores, factual_scores):
    """
    Youden's J (max(TPR-FPR)) threshold over the full ROC curve,
    treating label==1 (non-factual) as positive. This is the
    threshold SelfCheckGPT's OWN scores would use if forced to
    pick a single binary decision boundary that best separates
    the two classes on this dataset - a purely data-driven
    alternative to the hand-picked 0.5 default.
    """

    labels = [1] * len(nonfactual_scores) + [0] * len(factual_scores)
    scores = list(nonfactual_scores) + list(factual_scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    j = tpr - fpr

    best_idx = int(np.argmax(j))

    return float(thresholds[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])


def factual_calibrated_threshold(factual_scores, target_specificity=0.90):
    """
    The score below which target_specificity (e.g. 90%) of TRUE
    factual sentences fall. Using this as a subgroup cutoff asks:
    "which non-factual sentences does SelfCheckGPT score in the
    range typical of genuinely factual sentences?" - a direct,
    absolute operationalization of "confidently believed hallucination"
    that is anchored to the factual distribution rather than an
    arbitrary constant or an internal percentile of the non-factual
    distribution itself.
    """

    return float(np.percentile(np.array(factual_scores, dtype=float), target_specificity * 100))


def confusion_at_threshold(nonfactual_scores, factual_scores, threshold):

    nf = np.array(nonfactual_scores, dtype=float)
    f = np.array(factual_scores, dtype=float)

    tpr = float(np.mean(nf >= threshold)) if len(nf) else float("nan")  # recall on non-factual
    fpr = float(np.mean(f >= threshold)) if len(f) else float("nan")    # false-flag rate on factual

    return tpr, fpr


def plot_distributions(all_scores, factual_scores, nonfactual_scores, by_category, output_path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].hist(all_scores, bins=40, color="steelblue")
    axes[0, 0].set_title("All sentences")
    axes[0, 0].set_xlabel("SelfCheckGPT score")
    axes[0, 0].axvline(0.5, color="red", linestyle="--", linewidth=1)

    axes[0, 1].hist(factual_scores, bins=40, color="seagreen", alpha=0.7, label="factual")
    axes[0, 1].hist(nonfactual_scores, bins=40, color="firebrick", alpha=0.5, label="non-factual")
    axes[0, 1].set_title("Factual vs non-factual")
    axes[0, 1].set_xlabel("SelfCheckGPT score")
    axes[0, 1].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 1].legend()

    axes[1, 0].hist(nonfactual_scores, bins=40, color="firebrick")
    axes[1, 0].set_title("Non-factual sentences only")
    axes[1, 0].set_xlabel("SelfCheckGPT score")
    axes[1, 0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 0].axvline(median(nonfactual_scores), color="blue", linestyle=":", linewidth=1, label="median")
    axes[1, 0].legend()

    for cat, color in [("accurate", "seagreen"), ("minor_inaccurate", "orange"), ("major_inaccurate", "firebrick")]:
        vals = by_category.get(cat, [])
        if vals:
            axes[1, 1].hist(vals, bins=40, alpha=0.5, label=cat, color=color)
    axes[1, 1].set_title("By annotation category")
    axes[1, 1].set_xlabel("SelfCheckGPT score")
    axes[1, 1].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info("Wrote distribution plot to %s", output_path)


def main() -> None:

    dataset_name = "potsawee/wiki_bio_gpt3_hallucination"

    cache_path = PROJECT_ROOT / "experiments" / "results" / "selfcheck_cache.pkl"

    output_dir = PROJECT_ROOT / "experiments" / "results"

    logger.info("Loading dataset %s [evaluation]...", dataset_name)

    dataset = load_dataset(dataset_name)["evaluation"]

    logger.info("Loading SelfCheckGPT cache from %s...", cache_path)

    selfcheck_cache = load_selfcheck_cache(cache_path)

    scores, categories = collect_scores(dataset, selfcheck_cache)

    logger.info("Collected %d sentence-level scores from %d cached documents.", len(scores), len(selfcheck_cache))

    labels = [1 if LABEL_MAPPING[c] > 0.499 else 0 for c in categories]

    factual_scores = [s for s, l in zip(scores, labels) if l == 0]
    nonfactual_scores = [s for s, l in zip(scores, labels) if l == 1]

    by_category: dict[str, list[float]] = {"accurate": [], "minor_inaccurate": [], "major_inaccurate": []}
    for s, c in zip(scores, categories):
        by_category[c].append(s)

    # -----------------------------------------------------------------
    # 1. Overall / factual / non-factual distributions
    # -----------------------------------------------------------------

    logger.info("=" * 78)
    logger.info("1. SCORE DISTRIBUTIONS")
    logger.info("=" * 78)

    for name, vals in [
        ("all", scores),
        ("factual", factual_scores),
        ("non-factual", nonfactual_scores),
        ("accurate", by_category["accurate"]),
        ("minor_inaccurate", by_category["minor_inaccurate"]),
        ("major_inaccurate", by_category["major_inaccurate"]),
    ]:
        print_stats(describe(name, vals))

    # -----------------------------------------------------------------
    # 2. Bimodality check on the non-factual distribution
    # -----------------------------------------------------------------

    logger.info("=" * 78)
    logger.info("2. BIMODALITY CHECK (non-factual scores, GaussianMixture BIC)")
    logger.info("=" * 78)

    bimodal = bimodality_check(nonfactual_scores)

    logger.info(
        "BIC: 1-component=%.1f  2-component=%.1f  2-component wins=%s",
        bimodal["bic_1_component"], bimodal["bic_2_component"], bimodal["two_component_wins"],
    )
    logger.info(
        "2-component means=%s  weights=%s",
        [f"{m:.3f}" for m in bimodal["two_component_means"]],
        [f"{w:.3f}" for w in bimodal["two_component_weights"]],
    )

    # -----------------------------------------------------------------
    # 3. Is 0.5 meaningful? ROC-optimal and factual-calibrated thresholds
    # -----------------------------------------------------------------

    logger.info("=" * 78)
    logger.info("3. THRESHOLD ANALYSIS")
    logger.info("=" * 78)

    tpr_05, fpr_05 = confusion_at_threshold(nonfactual_scores, factual_scores, 0.5)

    logger.info(
        "At threshold=0.5: recall on non-factual=%.1f%%  false-flag rate on factual=%.1f%%",
        tpr_05 * 100, fpr_05 * 100,
    )
    logger.info(
        "=> %.1f%% of non-factual sentences ALREADY score >= 0.5 "
        "(i.e. 0.5 is not near the bulk of the non-factual distribution).",
        tpr_05 * 100,
    )

    roc_thresh, roc_tpr, roc_fpr = roc_optimal_threshold(nonfactual_scores, factual_scores)

    logger.info(
        "ROC-optimal (Youden's J) threshold=%.4f  (recall=%.1f%%, false-flag rate=%.1f%%)",
        roc_thresh, roc_tpr * 100, roc_fpr * 100,
    )

    calib_thresh_90 = factual_calibrated_threshold(factual_scores, target_specificity=0.90)
    calib_thresh_95 = factual_calibrated_threshold(factual_scores, target_specificity=0.95)

    tpr_90, fpr_90 = confusion_at_threshold(nonfactual_scores, factual_scores, calib_thresh_90)
    tpr_95, fpr_95 = confusion_at_threshold(nonfactual_scores, factual_scores, calib_thresh_95)

    logger.info(
        "Factual-calibrated threshold @90th pct of factual scores=%.4f "
        "(recall on non-factual=%.1f%%, false-flag rate=%.1f%%)",
        calib_thresh_90, tpr_90 * 100, fpr_90 * 100,
    )
    logger.info(
        "Factual-calibrated threshold @95th pct of factual scores=%.4f "
        "(recall on non-factual=%.1f%%, false-flag rate=%.1f%%)",
        calib_thresh_95, tpr_95 * 100, fpr_95 * 100,
    )

    # -----------------------------------------------------------------
    # 4. Candidate subgroup definitions - size and composition
    # -----------------------------------------------------------------

    logger.info("=" * 78)
    logger.info("4. CANDIDATE SUBGROUP DEFINITIONS")
    logger.info("=" * 78)

    median_nonfactual = median(nonfactual_scores)

    def subgroup_report(label, predicate):
        members = [s for s in nonfactual_scores if predicate(s)]
        pct_of_nonfactual = 100.0 * len(members) / len(nonfactual_scores) if nonfactual_scores else float("nan")
        mean_score = mean(members) if members else float("nan")
        logger.info(
            "%-55s n=%-4d (%.1f%% of non-factual)  mean_score=%.4f",
            label, len(members), pct_of_nonfactual, mean_score,
        )
        return members

    subgroup_report(
        "A: score < 0.5 (fixed)",
        lambda s: s < 0.5,
    )
    subgroup_report(
        f"B: score < ROC-optimal ({roc_thresh:.4f})",
        lambda s: s < roc_thresh,
    )
    subgroup_report(
        f"B': score < factual-calibrated @90th ({calib_thresh_90:.4f})",
        lambda s: s < calib_thresh_90,
    )
    subgroup_report(
        f"C: score < median of non-factual (current, {median_nonfactual:.4f})",
        lambda s: s < median_nonfactual,
    )
    subgroup_report(
        "C': bottom 25th percentile of non-factual",
        lambda s: s < np.percentile(nonfactual_scores, 25),
    )

    plot_distributions(
        scores, factual_scores, nonfactual_scores, by_category,
        output_dir / "selfcheck_score_distribution.png",
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
