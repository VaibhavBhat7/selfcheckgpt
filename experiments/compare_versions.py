"""
Compares two belief caches (e.g. the archived v1 baseline vs
the new v2 after the extraction/canonicalization/matching
improvements) on: extraction density, SUPPORT/ABSENT/
CONTRADICT rates, MatchTier distribution (including INVERSE -
direct attribution evidence for inverse/symmetric matching),
and average graph connectivity (edges/document). If a
``belief_stability_auc.csv`` sits next to either cache, its
rows are included in the printed report too.

Usage
-----
    python experiments/compare_versions.py \\
        --v1 experiments/results/v1_baseline/belief_cache.pkl \\
        --v2 experiments/results/belief_cache.pkl
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from belief_stability.belief_cache import BeliefCache
from belief_stability.matcher import BeliefMatcher
from belief_stability.matcher.semantic import SemanticMatcher
from belief_stability.models import PassageBeliefs
from belief_stability.scoring import BeliefPersistence, GraphScorer


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--v1", required=True, help="Path to the v1 (baseline) belief_cache.pkl")
    parser.add_argument("--v2", required=True, help="Path to the v2 (new) belief_cache.pkl")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "experiments" / "results" / "version_comparison.csv"),
    )

    return parser.parse_args()


def _summarize(cache: BeliefCache, cache_path: Path) -> Dict[str, float]:

    # NOTE (found during the Belief Stability module review): this
    # used to call BeliefMatcher() with no arguments, which silently
    # disables semantic matching (Tier 2) regardless of config,
    # making tier_semantic_rate read as 0.0 even when semantic
    # matching is genuinely contributing (measured ~2.2% of all
    # transitions via experiments/measure_match_tiers.py, which wires
    # a real SemanticMatcher). Loading each cache's own embeddings
    # file here fixes that confound.
    embeddings_path = cache_path.with_suffix(cache_path.suffix + ".embeddings.pkl")

    semantic_matcher = SemanticMatcher.load(embeddings_path) if embeddings_path.exists() else None

    matcher = BeliefMatcher(semantic_matcher=semantic_matcher)

    persistence = BeliefPersistence()

    num_docs = len(cache)

    original_belief_count = 0

    tier_counts: Counter = Counter()

    transition_counts: Counter = Counter()

    total_edges = 0

    for entry in cache.entries.values():

        all_original = [b for beliefs in entry.original_beliefs for b in beliefs]

        original_belief_count += len(all_original)

        original_passage = PassageBeliefs(passage_id=0, beliefs=all_original)

        sampled_passages = [
            PassageBeliefs(passage_id=idx + 1, beliefs=beliefs)
            for idx, beliefs in enumerate(entry.sampled_beliefs)
        ]

        transition_results = []

        for sampled_passage in sampled_passages:
            transition_results.extend(matcher.match_all(original_passage, sampled_passage))

        for result in transition_results:
            tier_counts[result.tier.value] += 1
            transition_counts[result.transition.value] += 1

        profiles = persistence.compute(transition_results)

        adjacency = GraphScorer._build_adjacency(profiles)

        total_edges += sum(len(neighbors) for neighbors in adjacency.values()) // 2

    total_transitions = sum(transition_counts.values()) or 1

    return {
        "num_docs": num_docs,
        "avg_original_beliefs_per_doc": original_belief_count / max(num_docs, 1),
        "support_rate": transition_counts["support"] / total_transitions,
        "absent_rate": transition_counts["absent"] / total_transitions,
        "contradict_rate": transition_counts["contradict"] / total_transitions,
        "tier_exact_rate": tier_counts["exact"] / total_transitions,
        "tier_inverse_rate": tier_counts["inverse"] / total_transitions,
        "tier_semantic_rate": tier_counts["semantic"] / total_transitions,
        "tier_nli_rate": tier_counts["nli"] / total_transitions,
        "tier_none_rate": tier_counts["none"] / total_transitions,
        "avg_graph_edges_per_doc": total_edges / max(num_docs, 1),
    }


def _load_auc_table(cache_path: Path) -> list[dict]:

    csv_path = cache_path.parent / "belief_stability_auc.csv"

    if not csv_path.exists():
        return []

    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:

    args = parse_args()

    v1_path = Path(args.v1)

    v2_path = Path(args.v2)

    print(f"Loading v1 cache from {v1_path}...")

    v1_cache = BeliefCache.load(v1_path)

    print(f"Loading v2 cache from {v2_path}...")

    v2_cache = BeliefCache.load(v2_path)

    v1_summary = _summarize(v1_cache, v1_path)

    v2_summary = _summarize(v2_cache, v2_path)

    fieldnames = ["metric", "v1", "v2", "delta"]

    rows = []

    for key in v1_summary:

        v1_value = v1_summary[key]

        v2_value = v2_summary[key]

        rows.append({
            "metric": key,
            "v1": round(v1_value, 4),
            "v2": round(v2_value, 4),
            "delta": round(v2_value - v1_value, 4),
        })

    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote comparison table to {output_path}\n")

    for row in rows:
        print(f"{row['metric']:32s}  v1={row['v1']:>10}  v2={row['v2']:>10}  delta={row['delta']:>10}")

    v1_auc = _load_auc_table(v1_path)

    v2_auc = _load_auc_table(v2_path)

    if v1_auc or v2_auc:

        print("\nBelief Stability AUC-PR (from belief_stability_auc.csv next to each cache):")
        print("v1:", v1_auc)
        print("v2:", v2_auc)


if __name__ == "__main__":
    main()
