"""
Online evaluation harness for Belief Stability against the
wiki_bio_gpt3_hallucination human annotations.

Requires a belief cache already built by
experiments/build_belief_cache.py - this script never loads
REBEL or any GPU model; it only does dict lookups and small
numpy ops over the cached beliefs, so re-running it (e.g. to
compare scoring methods) takes seconds.

Produces an ablation table (baseline vs bayesian vs graph)
plus PR-curve plots, written to experiments/results/.

Score polarity: BeliefStabilityResult.stability_score is in
[-1, 1], where +1 = fully supported (likely factual) and -1
= fully contradicted (likely hallucinated). This is the
OPPOSITE polarity from SelfCheckGPT-NLI's P(contradiction)
score (higher = more hallucinated), so "detect False"
(non-factual) uses -stability_score and "detect True"
(factual) uses stability_score directly.

Usage
-----
    python experiments/evaluate_belief.py
    python experiments/evaluate_belief.py --methods bayesian graph
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from belief_stability.belief_cache import BeliefCache
from belief_stability.config import BeliefStabilityConfig
from belief_stability.evaluation import (
    build_human_labels,
    compute_auc,
    get_pr_with_human_labels,
    random_baseline,
)
from belief_stability.matcher.semantic import SemanticMatcher
from belief_stability.pipeline import BeliefStabilityPipeline
from belief_stability.sentence_runner import SentenceBeliefRunner
from belief_stability.utils import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset", default="potsawee/wiki_bio_gpt3_hallucination"
    )
    parser.add_argument("--split", default="evaluation")
    parser.add_argument(
        "--cache",
        default=str(PROJECT_ROOT / "experiments" / "results" / "belief_cache.pkl"),
    )
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "default.yaml")
    )
    parser.add_argument(
        "--methods", nargs="+", default=["baseline", "bayesian", "graph"]
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments" / "results"),
    )
    parser.add_argument(
        "--use-nli-arbitration", action="store_true",
        help="Override config.use_nli_arbitration=True for this run (isolated experiment flag, "
             "does not touch configs/default.yaml).",
    )

    return parser.parse_args()


def evaluate_method(
    method: str,
    dataset,
    cache: BeliefCache,
    config: BeliefStabilityConfig,
    semantic_matcher: SemanticMatcher,
    nli_arbitrator=None,
) -> dict[int, list[float]]:

    method_config = BeliefStabilityConfig(**{**config.__dict__, "scoring_method": method})

    pipeline = BeliefStabilityPipeline(
        config=method_config,
        semantic_matcher=semantic_matcher if method_config.use_semantic_matching else None,
        nli_arbitrator=nli_arbitrator if method_config.use_nli_arbitration else None,
    )

    runner = SentenceBeliefRunner(pipeline=pipeline)

    belief_scores: dict[int, list[float]] = {}

    for i in range(len(dataset)):

        x = dataset[i]

        idx = x["wiki_bio_test_idx"]

        if not cache.has(idx):
            logger.warning("No cached beliefs for dataset id %s, skipping.", idx)
            continue

        entry = cache.get(idx)

        belief_scores[idx] = runner.run_from_cache(
            original_beliefs=entry.original_beliefs,
            sampled_beliefs=entry.sampled_beliefs,
            aggregation=method_config.sentence_aggregation,
        )

    return belief_scores


def main() -> None:

    args = parse_args()

    config = BeliefStabilityConfig.from_yaml(args.config)

    if args.use_nli_arbitration:
        config.use_nli_arbitration = True

    logger.info("Loading dataset %s [%s]...", args.dataset, args.split)

    dataset = load_dataset(args.dataset)[args.split]

    logger.info("Loading belief cache from %s...", args.cache)

    cache = BeliefCache.load(args.cache)

    embeddings_path = Path(args.cache).with_suffix(Path(args.cache).suffix + ".embeddings.pkl")

    semantic_matcher = (
        SemanticMatcher.load(embeddings_path)
        if embeddings_path.exists()
        else SemanticMatcher()
    )

    nli_arbitrator = None

    if config.use_nli_arbitration:
        from belief_stability.matcher.nli_arbitrator import NLIArbitrator
        logger.info("Loading NLIArbitrator (DeBERTa-v3-large-MNLI) for Tier 3 matching...")
        nli_arbitrator = NLIArbitrator()

    detect_false, detect_true, detect_false_hard = build_human_labels(dataset)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for method in args.methods:

        logger.info("Evaluating scoring method: %s", method)

        belief_scores = evaluate_method(method, dataset, cache, config, semantic_matcher, nli_arbitrator)

        hallucination_scores = {
            idx: [-s for s in scores] for idx, scores in belief_scores.items()
        }

        precision, recall = get_pr_with_human_labels(
            hallucination_scores, detect_false, pos_label=1
        )
        auc_false = compute_auc(precision, recall)

        precision_h, recall_h = get_pr_with_human_labels(
            hallucination_scores, detect_false_hard, pos_label=1
        )
        auc_false_hard = compute_auc(precision_h, recall_h)

        precision_t, recall_t = get_pr_with_human_labels(
            belief_scores, detect_true, pos_label=1
        )
        auc_true = compute_auc(precision_t, recall_t)

        logger.info(
            "%s: NonFact AUC-PR=%.2f, NonFact* AUC-PR=%.2f, Factual AUC-PR=%.2f",
            method, auc_false, auc_false_hard, auc_true,
        )

        rows.append({
            "method": method,
            "nonfact_auc_pr": round(auc_false, 2),
            "nonfact_hard_auc_pr": round(auc_false_hard, 2),
            "factual_auc_pr": round(auc_true, 2),
            "random_baseline_nonfact": round(random_baseline(detect_false) * 100, 2),
        })

        _plot_pr_curve(
            recall, precision,
            title=f"Belief Stability ({method}) - Detect NonFactual",
            output_path=output_dir / f"pr_curve_{method}_nonfact.png",
        )

    csv_path = output_dir / "belief_stability_auc.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote ablation table to %s", csv_path)


def _plot_pr_curve(recall, precision, title: str, output_path: Path) -> None:

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5.5, 4.5))
    plt.plot(recall, precision, "-")
    plt.ylim(0.0, 1.02)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
