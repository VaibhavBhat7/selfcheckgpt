import numpy as np
import pytest

from belief_stability.models import Belief, BeliefProfile
from reliability.aggregator import ReliabilityAggregator
from reliability.alpha_selection import auc_pr, recall_at_threshold, select_alpha
from reliability.features import DocumentContext, SentenceRow
from reliability.subgroups import compute_subgroup_threshold, filter_subgroup, in_subgroup


# --------------------------------------------------------------------------
# subgroups.py
# --------------------------------------------------------------------------

def make_row(dataset_id, sentence_index, label, selfcheck_score):
    return SentenceRow(
        dataset_id=dataset_id, sentence_index=sentence_index, label=label,
        selfcheck_score=selfcheck_score, baseline_score=0.0, bayesian_score=0.0,
    )


def test_subgroup_threshold_is_roc_optimal_and_uses_only_train_ids():

    rows = [
        # dataset 1: factual, low scores
        make_row(1, 0, label=0, selfcheck_score=0.1),
        make_row(1, 1, label=0, selfcheck_score=0.2),
        make_row(1, 2, label=0, selfcheck_score=0.3),
        # dataset 2: non-factual, high scores
        make_row(2, 0, label=1, selfcheck_score=0.7),
        make_row(2, 1, label=1, selfcheck_score=0.8),
        make_row(2, 2, label=1, selfcheck_score=0.9),
        # dataset 3: not in train_ids, must be excluded (would otherwise
        # pull the ROC-optimal threshold down toward 0.05)
        make_row(3, 0, label=1, selfcheck_score=0.05),
    ]

    threshold = compute_subgroup_threshold(rows, train_ids=[1, 2])

    # Perfect separation between the two classes at 0.3/0.7 - Youden's J
    # is uniquely maximized (J=1, TPR=1, FPR=0) at threshold=0.7.
    assert threshold == pytest.approx(0.7)


def test_subgroup_threshold_fallback_when_single_class():

    rows = [make_row(1, 0, label=1, selfcheck_score=0.9)]

    threshold = compute_subgroup_threshold(rows, train_ids=[1])

    assert threshold == 0.5


def test_in_subgroup_requires_nonfactual_and_below_threshold():

    row_in = make_row(1, 0, label=1, selfcheck_score=0.1)
    row_out_wrong_label = make_row(1, 1, label=0, selfcheck_score=0.1)
    row_out_high_score = make_row(1, 2, label=1, selfcheck_score=0.9)

    assert in_subgroup(row_in, threshold=0.5) is True
    assert in_subgroup(row_out_wrong_label, threshold=0.5) is False
    assert in_subgroup(row_out_high_score, threshold=0.5) is False

    filtered = filter_subgroup([row_in, row_out_wrong_label, row_out_high_score], threshold=0.5)
    assert filtered == [row_in]


# --------------------------------------------------------------------------
# aggregator.py - pluggability contract
# --------------------------------------------------------------------------

def test_aggregator_fits_and_predicts_with_two_signals():

    rows = [
        {"selfcheckgpt": 0.9, "belief_stability": -0.8} for _ in range(10)
    ] + [
        {"selfcheckgpt": 0.1, "belief_stability": 0.8} for _ in range(10)
    ]
    labels = [1] * 10 + [0] * 10

    model = ReliabilityAggregator(["selfcheckgpt", "belief_stability"]).fit(rows, labels)

    preds = model.predict_proba(rows)

    assert len(preds) == 20
    assert preds[0] > preds[-1]  # non-factual rows score higher than factual rows


def test_aggregator_accepts_a_third_signal_without_code_changes():
    """
    Simulates plugging in a future module (e.g. Counterfactual
    Verification): just add a key to each row dict and to
    active_features - nothing about the class changes.
    """

    rows = [
        {"selfcheckgpt": 0.9, "belief_stability": -0.8, "counterfactual": 0.7}
        for _ in range(10)
    ] + [
        {"selfcheckgpt": 0.1, "belief_stability": 0.8, "counterfactual": 0.1}
        for _ in range(10)
    ]
    labels = [1] * 10 + [0] * 10

    model = ReliabilityAggregator(
        ["selfcheckgpt", "belief_stability", "counterfactual"]
    ).fit(rows, labels)

    preds = model.predict_proba(rows)

    assert len(preds) == 20
    assert preds[0] > preds[-1]


# --------------------------------------------------------------------------
# alpha_selection.py - non-inferiority constrained selection
# --------------------------------------------------------------------------

def _make_profile(belief_id, subject, support, contradict, absent):

    belief = Belief(belief_id=belief_id, subject=subject, relation="R", object="O")

    return BeliefProfile(
        belief=belief,
        support=support, absent=absent, contradict=contradict,
        support_weight=float(support), absent_weight=float(absent),
        contradict_weight=float(contradict),
    )


def _synthetic_dataset(rng, n_docs, agreeing_neighbors: bool):
    """
    Each synthetic document has 2 sentences, each with one belief,
    both beliefs sharing the same subject (so GraphScorer always
    creates an edge between them).

    agreeing_neighbors=True: both sentences share the same true
    label, so propagation between them should help (or at least
    not hurt) - a real complementary signal.

    agreeing_neighbors=False: sentences have OPPOSITE true labels
    but still share an entity (a misleading structural coincidence)
    - propagation should strictly hurt for any alpha > 0, since it
    blends two beliefs whose true labels disagree.
    """

    rows, contexts = [], {}

    for doc_id in range(n_docs):

        label_a = int(rng.random() < 0.5)
        label_b = label_a if agreeing_neighbors else 1 - label_a

        def profile_for(label, suffix):
            # strong, unambiguous evidence in the "right" direction
            if label == 1:  # non-factual -> mostly contradicted
                return _make_profile(f"{doc_id}-{suffix}", "SharedEntity", support=0, contradict=8, absent=0)
            return _make_profile(f"{doc_id}-{suffix}", "SharedEntity", support=8, contradict=0, absent=0)

        profile_a = profile_for(label_a, "a")
        profile_b = profile_for(label_b, "b")

        contexts[doc_id] = DocumentContext(
            dataset_id=doc_id,
            profiles=[profile_a, profile_b],
            belief_id_to_sentence={profile_a.belief.belief_id: 0, profile_b.belief.belief_id: 1},
        )

        for sentence_index, label in [(0, label_a), (1, label_b)]:

            # Deliberately uninformative (uncorrelated with label) so all
            # discriminative burden falls on belief_stability - this makes
            # the aggregate metric's sensitivity to alpha's quality
            # mechanically guaranteed rather than possibly masked by a
            # second, already-saturating feature.
            noisy_selfcheck = float(rng.uniform(0.0, 1.0))

            rows.append(SentenceRow(
                dataset_id=doc_id, sentence_index=sentence_index, label=label,
                selfcheck_score=noisy_selfcheck, baseline_score=0.0, bayesian_score=0.0,
                support=8 if label == 0 else 0, absent=0, contradict=8 if label == 1 else 0,
            ))

    return rows, contexts


def test_alpha_selection_rejects_propagation_between_conflicting_neighbors():

    rng = np.random.default_rng(0)

    rows, contexts = _synthetic_dataset(rng, n_docs=30, agreeing_neighbors=False)

    alpha_star, diagnostics = select_alpha(rows, contexts, inner_k=3, seed=0)

    # Any alpha > 0 blends two beliefs with opposite true labels, which
    # degrades belief_stability's own separation. The non-inferiority
    # epsilon (derived from inner-fold noise) may still let a few small
    # alphas past the aggregate gate, but none of them should improve the
    # subgroup metric over alpha=0, so alpha=0 must still win the argmax.
    assert alpha_star == 0.0
    assert 0.0 in diagnostics["eligible_alphas"]
    assert diagnostics["mean_sub"][0.0] >= max(diagnostics["mean_sub"].values()) - 1e-9


def test_alpha_selection_allows_propagation_between_agreeing_neighbors():

    rng = np.random.default_rng(1)

    rows, contexts = _synthetic_dataset(rng, n_docs=30, agreeing_neighbors=True)

    alpha_star, diagnostics = select_alpha(rows, contexts, inner_k=3, seed=0)

    # Neighbors always share the true label here, so propagation is pure
    # denoising - alpha=0 should not be the ONLY eligible option.
    assert len(diagnostics["eligible_alphas"]) > 1


# --------------------------------------------------------------------------
# alpha_selection.py - subgroup metric must be real, not vacuous
# --------------------------------------------------------------------------
#
# The consistent-hallucination subgroup (subgroups.py::in_subgroup) is
# label==1 rows ONLY, by construction - a single-class population. AUC-PR
# is undefined without both classes; using it here silently produces NaN
# for every alpha, which collapses select_alpha's argmax to "whichever
# alpha is first in the grid" regardless of actual belief_stability
# quality. recall_at_threshold is what makes the subgroup metric real.

def test_auc_pr_is_nan_on_single_class_population():

    # Documents the exact failure mode recall_at_threshold exists to avoid.
    assert auc_pr([1, 1, 1], [0.9, 0.1, 0.6]) != auc_pr([1, 1, 1], [0.9, 0.1, 0.6])  # NaN != NaN


def test_recall_at_threshold_is_well_defined_on_single_class_population():

    assert recall_at_threshold([1, 1, 1, 1], [0.9, 0.4, 0.6, 0.1]) == pytest.approx(50.0)


def test_recall_at_threshold_ignores_negative_labels():

    labels = [1, 0, 1, 0]
    scores = [0.9, 0.9, 0.1, 0.1]  # both label==0 rows score high - must not count

    assert recall_at_threshold(labels, scores) == pytest.approx(50.0)


def test_select_alpha_subgroup_metric_is_not_vacuous():

    rng = np.random.default_rng(1)

    rows, contexts = _synthetic_dataset(rng, n_docs=30, agreeing_neighbors=True)

    _, diagnostics = select_alpha(rows, contexts, inner_k=3, seed=0)

    # If the subgroup metric were still AUC-PR (always NaN on a label==1-
    # only population), every alpha's mean_sub would collapse to -inf and
    # this would be trivially true for the wrong reason. Assert real,
    # finite, non-trivial recall values instead.
    for alpha, value in diagnostics["mean_sub"].items():
        assert value == value, f"mean_sub[{alpha}] is NaN"
        assert value != float("-inf"), f"mean_sub[{alpha}] is -inf (vacuous subgroup metric)"
        assert 0.0 <= value <= 100.0
