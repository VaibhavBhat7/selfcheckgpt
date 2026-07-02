"""
One-off analysis script: tabulates raw (pre-mapping) relation
string frequencies across a raw claim cache, so RELATION_MAPPING
expansions (belief_stability/constants.py) can be curated from
actual observed REBEL output rather than guessed. Run by hand,
not part of the regular pipeline.

Usage
-----
    python experiments/relation_frequency_report.py
    python experiments/relation_frequency_report.py --top 60
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from belief_stability.belief_cache import ClaimCache
from belief_stability.canonicalizer.relation_mapper import RelationMapper
from belief_stability.constants import RELATION_MAPPING


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--claim-cache",
        default=str(PROJECT_ROOT / "experiments" / "results" / "claim_cache.pkl"),
    )
    parser.add_argument("--top", type=int, default=40)

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    claim_cache = ClaimCache.load(args.claim_cache)

    mapper = RelationMapper()

    counter: Counter = Counter()

    total = 0

    for example in claim_cache.entries.values():

        for claims in (*example.original_claims, *example.sampled_claims):

            for claim in claims:

                normalized = mapper._normalize_format(claim.relation)

                counter[normalized] += 1

                total += 1

    mapped = {r: c for r, c in counter.items() if r in RELATION_MAPPING}

    unmapped = {r: c for r, c in counter.items() if r not in RELATION_MAPPING}

    mapped_total = sum(mapped.values())

    unmapped_total = sum(unmapped.values())

    print(f"Total raw claims: {total}")
    print(f"Mapped (covered by RELATION_MAPPING): {mapped_total} ({100 * mapped_total / total:.1f}%)")
    print(f"Unmapped (falls through to .upper()): {unmapped_total} ({100 * unmapped_total / total:.1f}%)")
    print(f"Distinct unmapped raw relations: {len(unmapped)}")
    print()
    print(f"Top {args.top} unmapped relations by frequency:")

    for relation, count in sorted(unmapped.items(), key=lambda x: -x[1])[: args.top]:
        print(f"{count:6d}  {relation}")


if __name__ == "__main__":
    main()
