"""
---------------------------------------------------------
Reliability Aggregation Engine

File        : features.py

Description
-----------
Builds the per-sentence feature table used by the fusion CV
harness, from the existing (Phase 1) BeliefCache plus a
SelfCheckGPT-NLI score cache.

Important scoping decision: matching is done ONCE PER
DOCUMENT (all of a document's original-sentence beliefs
together), not once per sentence. Baseline/Bayesian scores
are per-belief-independent, so this doesn't change their
values versus sentence-scoped matching - but it does let
GraphScorer see entity-sharing edges ACROSS sentences of the
same passage (e.g. two different sentences both about "Steve
Jobs"), which a sentence-scoped graph could never have. Per-
belief profiles are grouped back into per-sentence rows via
belief_id -> sentence_index (Belief.belief_id is a random
uuid assigned at cache-build time, so this is unambiguous).

DocumentContext keeps the whole-document profiles around so
the alpha sweep (alpha_selection.py) can recompute
GraphScorer at many alpha values without re-matching.
---------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from belief_stability.belief_cache import BeliefCache
from belief_stability.config import BeliefStabilityConfig
from belief_stability.evaluation import LABEL_MAPPING
from belief_stability.matcher import BeliefMatcher
from belief_stability.matcher.semantic import SemanticMatcher
from belief_stability.models import Belief, BeliefProfile, PassageBeliefs
from belief_stability.scoring import BaselineScorer, BayesianScorer, BeliefPersistence, GraphScorer


@dataclass
class SentenceRow:

    dataset_id: int

    sentence_index: int

    label: int  # 1 = non-factual (raw_label > 0.499), 0 = factual

    selfcheck_score: float

    baseline_score: float

    bayesian_score: float

    support: float = 0.0

    absent: float = 0.0

    contradict: float = 0.0


@dataclass
class DocumentContext:

    dataset_id: int

    profiles: List[BeliefProfile] = field(default_factory=list)

    belief_id_to_sentence: Dict[str, int] = field(default_factory=dict)


def build_dataset(
    dataset,
    belief_cache: BeliefCache,
    selfcheck_cache: Dict[int, List[float]],
    config: BeliefStabilityConfig,
    semantic_matcher: SemanticMatcher | None = None,
    nli_arbitrator=None,
) -> Tuple[List[SentenceRow], Dict[int, DocumentContext]]:
    """
    Returns (rows, contexts). ``rows`` is a flat per-sentence
    table ready for the aggregator (missing only the alpha-
    dependent graph score, attached later via
    ``attach_graph_scores``). ``contexts`` holds the whole-
    document profiles needed to compute that graph score at
    any alpha.

    ``nli_arbitrator``: optional ``NLIArbitrator`` instance, only
    consulted when ``config.use_nli_arbitration`` is True (matches
    the existing pattern for ``semantic_matcher``/``use_semantic_matching``).
    Caller is responsible for constructing it (it loads a transformer
    onto a device) - passing None here always disables Tier 3
    regardless of the config flag, so existing callers that don't
    pass it keep their current behavior unchanged.
    """

    matcher = BeliefMatcher(
        semantic_matcher=semantic_matcher if config.use_semantic_matching else None,
        nli_arbitrator=nli_arbitrator if config.use_nli_arbitration else None,
        match_similarity_threshold=config.match_similarity_threshold,
        nli_ambiguous_low=config.nli_ambiguous_low,
        nli_ambiguous_high=config.nli_ambiguous_high,
        use_inverse_matching=config.use_inverse_matching,
    )

    persistence = BeliefPersistence()

    baseline_scorer = BaselineScorer()

    bayesian_scorer = BayesianScorer(absent_discount=config.absent_discount)

    rows: List[SentenceRow] = []

    contexts: Dict[int, DocumentContext] = {}

    for i in range(len(dataset)):

        x = dataset[i]

        dataset_id = x["wiki_bio_test_idx"]

        if not belief_cache.has(dataset_id) or dataset_id not in selfcheck_cache:
            continue

        entry = belief_cache.get(dataset_id)

        selfcheck_scores = selfcheck_cache[dataset_id]

        raw_labels = [LABEL_MAPPING[a] for a in x["annotation"]]

        all_beliefs: List[Belief] = []

        belief_id_to_sentence: Dict[str, int] = {}

        for sentence_index, sentence_beliefs in enumerate(entry.original_beliefs):

            for belief in sentence_beliefs:
                belief_id_to_sentence[belief.belief_id] = sentence_index

            all_beliefs.extend(sentence_beliefs)

        original_passage = PassageBeliefs(passage_id=0, beliefs=all_beliefs)

        sampled_passages = [
            PassageBeliefs(passage_id=idx + 1, beliefs=beliefs)
            for idx, beliefs in enumerate(entry.sampled_beliefs)
        ]

        transition_results = []

        for sampled_passage in sampled_passages:
            transition_results.extend(matcher.match_all(original_passage, sampled_passage))

        profiles = persistence.compute(transition_results)

        contexts[dataset_id] = DocumentContext(
            dataset_id=dataset_id,
            profiles=profiles,
            belief_id_to_sentence=belief_id_to_sentence,
        )

        per_sentence_profiles: Dict[int, List[BeliefProfile]] = defaultdict(list)

        for profile in profiles:

            sentence_index = belief_id_to_sentence.get(profile.belief.belief_id)

            if sentence_index is not None:
                per_sentence_profiles[sentence_index].append(profile)

        num_sentences = min(
            len(entry.original_beliefs),
            len(x["gpt3_sentences"]),
            len(selfcheck_scores),
            len(raw_labels),
        )

        for sentence_index in range(num_sentences):

            sentence_profiles = per_sentence_profiles.get(sentence_index, [])

            baseline_score = baseline_scorer.compute(sentence_profiles).stability_score

            bayesian_score = bayesian_scorer.compute(sentence_profiles).stability_score

            rows.append(
                SentenceRow(
                    dataset_id=dataset_id,
                    sentence_index=sentence_index,
                    label=int(raw_labels[sentence_index] > 0.499),
                    selfcheck_score=float(selfcheck_scores[sentence_index]),
                    baseline_score=baseline_score,
                    bayesian_score=bayesian_score,
                    support=sum(p.support for p in sentence_profiles),
                    absent=sum(p.absent for p in sentence_profiles),
                    contradict=sum(p.contradict for p in sentence_profiles),
                )
            )

    return rows, contexts


def graph_scores_for_document(
    context: DocumentContext,
    alpha: float,
    absent_discount: float = 0.5,
) -> Dict[int, float]:
    """
    Score the WHOLE document's belief graph at a given alpha,
    then group the resulting per-belief scores back into a
    per-sentence mean. Cheap: reuses ``context.profiles``
    (already matched), no REBEL / re-matching involved.
    """

    scorer = GraphScorer(alpha=alpha, absent_discount=absent_discount)

    result = scorer.compute(context.profiles)

    per_sentence: Dict[int, List[float]] = defaultdict(list)

    for profile in result.profiles:

        sentence_index = context.belief_id_to_sentence.get(profile.belief.belief_id)

        if sentence_index is not None:
            per_sentence[sentence_index].append(profile.graph_score)

    return {
        sentence_index: sum(scores) / len(scores)
        for sentence_index, scores in per_sentence.items()
    }


def attach_graph_scores(
    rows: Sequence[SentenceRow],
    contexts: Dict[int, DocumentContext],
    alpha: float,
    absent_discount: float = 0.5,
) -> Dict[Tuple[int, int], float]:
    """
    Returns {(dataset_id, sentence_index): graph_score}.
    Computed once per document (not once per row) - documents
    appear multiple times in ``rows`` (once per sentence) but
    only need one graph pass each.
    """

    per_document_cache: Dict[int, Dict[int, float]] = {}

    scores: Dict[Tuple[int, int], float] = {}

    for row in rows:

        if row.dataset_id not in per_document_cache:

            per_document_cache[row.dataset_id] = graph_scores_for_document(
                contexts[row.dataset_id], alpha=alpha, absent_discount=absent_discount,
            )

        scores[(row.dataset_id, row.sentence_index)] = per_document_cache[row.dataset_id].get(
            row.sentence_index, 0.0
        )

    return scores
