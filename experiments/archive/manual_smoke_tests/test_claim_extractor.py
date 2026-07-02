"""
Test script for the REBEL Claim Extractor.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from belief_stability.claim_extractor import RebelClaimExtractor


def print_claims(title, claims):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if len(claims) == 0:

        print("No claims extracted.")
        return

    for i, claim in enumerate(claims, start=1):

        print(f"\nClaim {i}")

        print(f"Subject   : {claim.subject}")

        print(f"Relation  : {claim.relation}")

        print(f"Object    : {claim.object}")

        print(f"Attributes: {claim.attributes}")

        print("-" * 60)


def main():

    extractor = RebelClaimExtractor()

    sentences = [

        "Steve Jobs founded Apple in 1976.",

        "Lionel Messi plays for Inter Miami.",

        "Barack Obama was born in Hawaii.",

        "Sundar Pichai is the CEO of Google.",

        "MS Dhoni captained India and won the 2011 Cricket World Cup."

    ]

    for sentence in sentences:

        claims = extractor.extract(sentence)

        print_claims(sentence, claims)


if __name__ == "__main__":

    main()