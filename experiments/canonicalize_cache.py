"""
Fast, CPU-only stage: turns a raw claim cache
(experiments/build_belief_cache.py output) into a
canonicalized belief cache, ready for evaluate_belief.py /
evaluate_reliability.py. Re-runs in low minutes - no REBEL,
no GPU - so this is the step to re-run any time
canonicalization/matching config changes (RELATION_MAPPING,
entity normalization, inverse matching).

Usage
-----
    python experiments/canonicalize_cache.py
        # full run: claim_cache.pkl -> belief_cache.pkl

    python experiments/canonicalize_cache.py --limit 5

    python experiments/canonicalize_cache.py \\
        --output experiments/results/belief_cache_no_entity_norm.pkl \\
        --disable-document-entity-normalization
        # ablation run, isolates point 3's contribution

    python experiments/canonicalize_cache.py \\
        --output experiments/results/belief_cache_no_inverse.pkl \\
        --disable-inverse-matching
        # ablation run, isolates point 5's contribution
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from belief_stability.belief_cache import ClaimCache
from belief_stability.canonicalize_cache import CacheCanonicalizer
from belief_stability.config import BeliefStabilityConfig
from belief_stability.utils import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--claim-cache",
        default=str(PROJECT_ROOT / "experiments" / "results" / "claim_cache.pkl"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "experiments" / "results" / "belief_cache.pkl"),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-document-entity-normalization", action="store_true")
    parser.add_argument("--disable-inverse-matching", action="store_true")

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    config = BeliefStabilityConfig.from_yaml(args.config)

    if args.disable_document_entity_normalization:
        config.use_document_entity_normalization = False

    if args.disable_inverse_matching:
        config.use_inverse_matching = False

    logger.info("Loading claim cache from %s...", args.claim_cache)

    claim_cache = ClaimCache.load(args.claim_cache)

    canonicalizer = CacheCanonicalizer(config=config)

    start = time.monotonic()

    belief_cache = canonicalizer.canonicalize_cache(claim_cache, limit=args.limit)

    output_path = Path(args.output)

    belief_cache.save(output_path)

    canonicalizer.build_embedding_cache(belief_cache, output_path)

    elapsed = time.monotonic() - start

    logger.info(
        "Done: %s examples canonicalized to %s in %.1fs.",
        len(belief_cache), output_path, elapsed,
    )


if __name__ == "__main__":
    main()
