"""
Offline stage: compute SelfCheckGPT-NLI scores for the whole
wiki_bio_gpt3_hallucination dataset ONCE and cache them to disk.

SelfCheckNLI.predict() (selfcheckgpt/modeling_selfcheck.py) is
unbatched - one DeBERTa-v3-large-MNLI forward pass per
(sentence, sampled_passage) pair, done in a nested Python loop.
That's third-party code and out of scope to rewrite here; instead
we pay the cost once and cache the result, exactly like
build_belief_cache.py does for REBEL.

Cache format: pickled dict {wiki_bio_test_idx: [score_per_sentence]}.

Usage
-----
    python experiments/build_selfcheck_cache.py --limit 5
        # smoke test: reports measured seconds/example before you
        # commit to the full ~60-90 minute run

    python experiments/build_selfcheck_cache.py --resume
        # full 238-example dataset, resumable if interrupted
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from datasets import load_dataset

from belief_stability.utils import setup_logger

logger = setup_logger(__name__)


def load_selfcheck_cache(path: str | Path) -> Dict[int, List[float]]:

    path = Path(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def save_selfcheck_cache(cache: Dict[int, List[float]], path: str | Path) -> None:

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset", default="potsawee/wiki_bio_gpt3_hallucination")
    parser.add_argument("--split", default="evaluation")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "experiments" / "results" / "selfcheck_cache.pkl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--device", default=None)

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    logger.info("Loading dataset %s [%s]...", args.dataset, args.split)

    dataset = load_dataset(args.dataset)[args.split]

    output_path = Path(args.output)

    cache: Dict[int, List[float]] = {}

    if args.resume and output_path.exists():
        cache = load_selfcheck_cache(output_path)
        logger.info("Resuming: %s examples already cached.", len(cache))

    logger.info("Loading SelfCheckNLI (DeBERTa-v3-large-MNLI) on %s...", device)

    selfcheck = SelfCheckNLI(device=device)

    num_examples = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    start_time = time.monotonic()

    processed = 0

    for i in range(num_examples):

        x = dataset[i]

        idx = x["wiki_bio_test_idx"]

        if idx in cache:
            continue

        scores = selfcheck.predict(
            sentences=x["gpt3_sentences"],
            sampled_passages=x["gpt3_text_samples"],
        )

        cache[idx] = scores.tolist()

        processed += 1

        if processed % args.flush_every == 0:

            save_selfcheck_cache(cache, output_path)

            elapsed = time.monotonic() - start_time

            rate = elapsed / processed

            logger.info(
                "Checkpointed %s examples (%.1f s/example so far).",
                len(cache), rate,
            )

    save_selfcheck_cache(cache, output_path)

    elapsed = time.monotonic() - start_time

    rate = elapsed / processed if processed else 0.0

    logger.info(
        "SelfCheck cache build complete: %s examples cached (%s newly processed) "
        "in %.1fs (%.1f s/example, including model load time).",
        len(cache), processed, elapsed, rate,
    )


if __name__ == "__main__":
    main()
