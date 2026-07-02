import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from belief_stability.sentence_runner import SentenceBeliefRunner


def main():

    runner = SentenceBeliefRunner()

    original_sentences = [

        "Steve Jobs founded Apple in 1976.",

        "Steve Jobs was the CEO of Apple.",

        "Apple is headquartered in Cupertino."

    ]

    sampled_passages = [

        """
        Steve Jobs founded Apple in 1976.
        Steve Jobs later became the CEO of Apple.
        Apple is headquartered in Cupertino.
        """,

        """
        Apple was founded by Steve Jobs.
        Steve Jobs served as Apple's CEO.
        Apple's headquarters are located in Cupertino.
        """,

        """
        Steve Jobs created Apple.
        Tim Cook is currently the CEO of Apple.
        Apple is based in Cupertino.
        """,

        """
        Steve Jobs worked at Apple.
        Apple began in 1976.
        Apple has its headquarters in Cupertino.
        """,

        """
        Steve Jobs founded Apple.
        Apple was established in 1976.
        Cupertino is Apple's headquarters.
        """

    ]

    print("\n")
    print("=" * 80)
    print("BELIEF STABILITY SENTENCE RUNNER")
    print("=" * 80)

    scores = runner.run_sentences(

        original_sentences=original_sentences,

        sampled_passages=sampled_passages,

    )

    print()

    for index, (sentence, score) in enumerate(

        zip(original_sentences, scores),

        start=1,

    ):

        print("-" * 80)

        print(f"Sentence {index}")

        print("-" * 80)

        print(sentence)

        print()

        print(f"Belief Stability Score : {score:.3f}")

        print()

    print("=" * 80)


if __name__ == "__main__":
    main()