"""
Offline stage: extract RAW claims (sentence-level, pronoun-
resolved) for the whole wiki_bio_gpt3_hallucination dataset
ONCE and cache them to disk. Canonicalization is a separate,
fast, CPU-only step - see experiments/canonicalize_cache.py.

Usage
-----
    python experiments/build_belief_cache.py --limit 5
        # smoke test: 5 examples, reports measured texts/sec
        # so you can extrapolate before committing to a full run

    python experiments/build_belief_cache.py --resume
        # full 238-example dataset, resumable if interrupted

    python experiments/build_belief_cache.py --resume --disable-pronoun-resolution \\
        --output experiments/results/claim_cache_no_coref.pkl
        # isolates extraction-density-only gains from pronoun
        # resolution's gains, for the attribution analysis
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset

from belief_stability.belief_cache import RawClaimCacheBuilder
from belief_stability.claim_extractor import RebelClaimExtractor
from belief_stability.config import BeliefStabilityConfig
from belief_stability.utils import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset", default="potsawee/wiki_bio_gpt3_hallucination"
    )
    parser.add_argument(
        "--split", default="evaluation"
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "experiments" / "results" / "claim_cache.pkl"),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--disable-pronoun-resolution", action="store_true",
        help="Build the raw cache without pronoun resolution, to isolate "
             "extraction-density-only gains from pronoun-resolution gains.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    config = BeliefStabilityConfig.from_yaml(args.config)

    logger.info("Loading dataset %s [%s]...", args.dataset, args.split)

    dataset = load_dataset(args.dataset)[args.split]

    logger.info("Loading REBEL extractor (batch_size=%s, precision=%s)...",
                config.extractor_batch_size, config.precision)

    extractor = RebelClaimExtractor(
        device=args.device,
        precision=config.precision,
        batch_size=config.extractor_batch_size,
        num_beams=config.num_beams,
        max_input_length=config.max_input_length,
        max_output_length=config.max_output_length,
    )

    use_pronoun_resolution = (
        config.use_pronoun_resolution and not args.disable_pronoun_resolution
    )

    builder = RawClaimCacheBuilder(
        extractor=extractor,
        use_pronoun_resolution=use_pronoun_resolution,
    )

    start = time.monotonic()

    cache = builder.build(
        dataset=dataset,
        output_path=args.output,
        flush_every=config.flush_every,
        resume=args.resume,
        limit=args.limit,
    )

    elapsed = time.monotonic() - start

    logger.info(
        "Done: %s examples cached to %s in %.1fs.",
        len(cache), args.output, elapsed,
    )


if __name__ == "__main__":
    main()
