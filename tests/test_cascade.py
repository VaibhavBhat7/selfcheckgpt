import pytest

from belief_stability.models import Belief, BeliefProfile
from reliability.cascade import (
    BeliefStabilityStage,
    CascadeStageOutput,
    ReliabilityCascade,
    SelfCheckGPTStage,
)
from reliability.features import DocumentContext, SentenceRow


def make_row(dataset_id, sentence_index, selfcheck_score, label=1):
    return SentenceRow(
        dataset_id=dataset_id, sentence_index=sentence_index, label=label,
        selfcheck_score=selfcheck_score, baseline_score=0.0, bayesian_score=0.0,
    )


def make_context(dataset_id, support=8, contradict=0, absent=0):

    belief = Belief(belief_id=f"{dataset_id}-b", subject="X", relation="R", object="Y")

    profile = BeliefProfile(
        belief=belief, support=support, absent=absent, contradict=contradict,
        support_weight=float(support), absent_weight=float(absent), contradict_weight=float(contradict),
    )

    return DocumentContext(
        dataset_id=dataset_id, profiles=[profile], belief_id_to_sentence={belief.belief_id: 0},
    )


# --------------------------------------------------------------------------
# SelfCheckGPTStage
# --------------------------------------------------------------------------

def test_selfcheckgpt_stage_resolves_at_or_above_threshold():

    stage = SelfCheckGPTStage(threshold=0.7)

    rows = [make_row(1, 0, selfcheck_score=0.9), make_row(1, 1, selfcheck_score=0.7)]

    outputs = stage.predict(rows)

    assert all(o.resolved for o in outputs)
    assert outputs[0].score == pytest.approx(0.9)
    assert outputs[1].score == pytest.approx(0.7)
    assert all(o.stage == "selfcheckgpt" for o in outputs)


def test_selfcheckgpt_stage_escalates_below_threshold():

    stage = SelfCheckGPTStage(threshold=0.7)

    rows = [make_row(1, 0, selfcheck_score=0.69), make_row(1, 1, selfcheck_score=0.1)]

    outputs = stage.predict(rows)

    assert all(not o.resolved for o in outputs)


# --------------------------------------------------------------------------
# BeliefStabilityStage
# --------------------------------------------------------------------------

def test_belief_stability_stage_is_terminal_by_default():

    context = make_context(1, support=8, contradict=0, absent=0)

    stage = BeliefStabilityStage(contexts={1: context}, alpha=0.3, absent_discount=0.5, blend_weight=0.5)

    rows = [make_row(1, 0, selfcheck_score=0.3)]

    outputs = stage.predict(rows)

    assert outputs[0].resolved is True
    assert outputs[0].stage == "belief_stability"


def test_belief_stability_stage_blends_selfcheck_and_belief_risk():

    # alpha=1+8=9, beta=1+0+0.5*0=1 -> posterior_mean=0.9 -> belief_score=0.8
    # no neighbors (single-node graph) -> graph_score == base_score == 0.8
    # risk = (1 - 0.8) / 2 = 0.1
    context = make_context(1, support=8, contradict=0, absent=0)

    stage = BeliefStabilityStage(contexts={1: context}, alpha=0.3, absent_discount=0.5, blend_weight=0.5)

    rows = [make_row(1, 0, selfcheck_score=0.4)]

    outputs = stage.predict(rows)

    expected = 0.5 * 0.4 + 0.5 * 0.1

    assert outputs[0].score == pytest.approx(expected, abs=1e-6)


def test_belief_stability_stage_not_terminal_can_escalate_further():

    context = make_context(1, support=8, contradict=0, absent=0)

    stage = BeliefStabilityStage(contexts={1: context}, alpha=0.3, is_terminal=False)

    rows = [make_row(1, 0, selfcheck_score=0.4)]

    outputs = stage.predict(rows)

    assert outputs[0].resolved is False


def test_belief_stability_stage_handles_empty_input():

    stage = BeliefStabilityStage(contexts={}, alpha=0.3)

    assert stage.predict([]) == []


# --------------------------------------------------------------------------
# ReliabilityCascade
# --------------------------------------------------------------------------

def test_cascade_resolves_confident_rows_at_selfcheckgpt_stage():

    context = make_context(1, support=0, contradict=8, absent=0)  # would score very "non-factual" if reached

    cascade = ReliabilityCascade([
        SelfCheckGPTStage(threshold=0.7),
        BeliefStabilityStage(contexts={1: context}, alpha=0.3, blend_weight=0.5, is_terminal=True),
    ])

    rows = [make_row(1, 0, selfcheck_score=0.95)]

    scores = cascade.predict(rows)

    # resolved at stage 1 -> raw selfcheck score used directly, stage 2 never invoked
    assert scores[(1, 0)] == pytest.approx(0.95)
    assert cascade.last_stage_assignment[(1, 0)] == "selfcheckgpt"


def test_cascade_escalates_unconfident_rows_to_belief_stability():

    # alpha=1+0=1, beta=1+8+0=9 -> posterior_mean=0.1 -> belief_score=-0.8 -> risk=0.9
    context = make_context(1, support=0, contradict=8, absent=0)

    cascade = ReliabilityCascade([
        SelfCheckGPTStage(threshold=0.7),
        BeliefStabilityStage(contexts={1: context}, alpha=0.3, blend_weight=0.5, is_terminal=True),
    ])

    rows = [make_row(1, 0, selfcheck_score=0.4)]

    scores = cascade.predict(rows)

    expected = 0.5 * 0.4 + 0.5 * 0.9

    assert scores[(1, 0)] == pytest.approx(expected, abs=1e-6)
    assert cascade.last_stage_assignment[(1, 0)] == "belief_stability"


def test_cascade_mixed_batch_routes_each_row_to_the_right_stage():

    context1 = make_context(1, support=8, contradict=0, absent=0)
    context2 = make_context(2, support=8, contradict=0, absent=0)

    cascade = ReliabilityCascade([
        SelfCheckGPTStage(threshold=0.7),
        BeliefStabilityStage(contexts={1: context1, 2: context2}, alpha=0.3, blend_weight=0.5, is_terminal=True),
    ])

    rows = [make_row(1, 0, selfcheck_score=0.95), make_row(2, 0, selfcheck_score=0.2)]

    scores = cascade.predict(rows)

    assert cascade.last_stage_assignment[(1, 0)] == "selfcheckgpt"
    assert cascade.last_stage_assignment[(2, 0)] == "belief_stability"
    assert scores[(1, 0)] == pytest.approx(0.95)


def test_cascade_raises_if_last_stage_does_not_resolve_everything():

    context = make_context(1, support=8, contradict=0, absent=0)

    cascade = ReliabilityCascade([
        SelfCheckGPTStage(threshold=0.7),
        BeliefStabilityStage(contexts={1: context}, alpha=0.3, is_terminal=False),  # never resolves
    ])

    rows = [make_row(1, 0, selfcheck_score=0.1)]

    with pytest.raises(RuntimeError):
        cascade.predict(rows)


def test_cascade_requires_at_least_one_stage():

    with pytest.raises(ValueError):
        ReliabilityCascade([])


def test_cascade_stage_output_dataclass_fields():

    output = CascadeStageOutput(score=0.5, resolved=True, stage="x")

    assert output.score == 0.5
    assert output.resolved is True
    assert output.stage == "x"
