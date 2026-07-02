"""
Read-only measurement of the REAL match-tier distribution using a
correctly-wired BeliefMatcher (real SemanticMatcher from the
embeddings cache, optionally a real NLIArbitrator) - as opposed to
experiments/compare_versions.py's ``_summarize()``, which
instantiates ``BeliefMatcher()`` with NO arguments and therefore
silently disables semantic + NLI matching regardless of config
(a confound discovered during the Belief Stability module review).

Usage
-----
    python experiments/measure_match_tiers.py                  # semantic on, NLI off (matches production default)
    python experiments/measure_match_tiers.py --with-nli        # semantic on, NLI on (ablation)
    python experiments/measure_match_tiers.py --limit 40        # smoke test on a subset (NLI is slow - one model call per ambiguous pair)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from belief_stability.belief_cache import BeliefCache
from belief_stability.config import BeliefStabilityConfig
from belief_stability.matcher import BeliefMatcher
from belief_stability.matcher.semantic import SemanticMatcher
from belief_stability.models import PassageBeliefs
from belief_stability.utils import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--belief-cache", default=str(PROJECT_ROOT / "experiments" / "results" / "belief_cache.pkl"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--with-nli", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    config = BeliefStabilityConfig.from_yaml(args.config)

    cache = BeliefCache.load(args.belief_cache)

    embeddings_path = Path(args.belief_cache).with_suffix(Path(args.belief_cache).suffix + ".embeddings.pkl")

    semantic_matcher = (
        SemanticMatcher.load(embeddings_path) if (config.use_semantic_matching and embeddings_path.exists()) else None
    )

    nli_arbitrator = None

    if args.with_nli:
        from belief_stability.matcher.nli_arbitrator import NLIArbitrator
        logger.info("Loading NLIArbitrator (DeBERTa-v3-large-MNLI)...")
        nli_arbitrator = NLIArbitrator()

    matcher = BeliefMatcher(
        semantic_matcher=semantic_matcher,
        nli_arbitrator=nli_arbitrator,
        match_similarity_threshold=config.match_similarity_threshold,
        nli_ambiguous_low=config.nli_ambiguous_low,
        nli_ambiguous_high=config.nli_ambiguous_high,
        use_inverse_matching=config.use_inverse_matching,
    )

    logger.info(
        "Matcher wired: semantic_matcher=%s (%d cached embeddings), nli_arbitrator=%s",
        semantic_matcher is not None, len(semantic_matcher.embedding_cache) if semantic_matcher else 0,
        nli_arbitrator is not None,
    )

    tier_counts: Counter = Counter()
    transition_counts: Counter = Counter()

    entries = list(cache.entries.values())
    if args.limit:
        entries = entries[: args.limit]

    start = time.monotonic()

    for i, entry in enumerate(entries):

        all_original = [b for beliefs in entry.original_beliefs for b in beliefs]
        original_passage = PassageBeliefs(passage_id=0, beliefs=all_original)

        sampled_passages = [
            PassageBeliefs(passage_id=idx + 1, beliefs=beliefs)
            for idx, beliefs in enumerate(entry.sampled_beliefs)
        ]

        for sampled_passage in sampled_passages:
            for result in matcher.match_all(original_passage, sampled_passage):
                tier_counts[result.tier.value] += 1
                transition_counts[result.transition.value] += 1

        if (i + 1) % 40 == 0:
            logger.info("Processed %d/%d docs (%.1fs elapsed)...", i + 1, len(entries), time.monotonic() - start)

    total = sum(transition_counts.values()) or 1

    logger.info("=" * 70)
    logger.info("Processed %d documents, %d total belief-transitions in %.1fs", len(entries), total, time.monotonic() - start)
    logger.info("=" * 70)
    logger.info("Transition rates: support=%.4f  absent=%.4f  contradict=%.4f",
                 transition_counts["support"] / total, transition_counts["absent"] / total, transition_counts["contradict"] / total)
    logger.info("Tier rates: exact=%.4f  inverse=%.4f  semantic=%.4f  nli=%.4f  none=%.4f",
                 tier_counts["exact"] / total, tier_counts["inverse"] / total, tier_counts["semantic"] / total,
                 tier_counts["nli"] / total, tier_counts["none"] / total)
    logger.info("Raw counts: %s", dict(tier_counts))


if __name__ == "__main__":
    main()
