"""
Reliability Aggregation Engine.

Fuses per-sentence confidence signals from independent verification
modules (SelfCheckGPT consistency today; Belief Stability; later
Counterfactual Verification and Adversarial Verification) into one
calibrated reliability score.

PRODUCTION DEFAULT: the multi-stage ``ReliabilityCascade`` (see
cascade.py) - trusts SelfCheckGPT when it's already confident (at or
above its own ROC-optimal threshold), escalates to Belief Stability
only when it isn't. Experimentally dominated the single global
logistic-regression fusion (``ReliabilityAggregator``, aggregator.py)
on both aggregate AUC-PR and consistent-hallucination-subgroup recall
(see experiments/compare_fusion_strategies.py). ``ReliabilityAggregator``
is kept for the bs_baseline/bs_bayesian/bs_graph comparison arms in
experiments/evaluate_reliability.py and as a documented alternative,
not as the recommended aggregation strategy. Counterfactual and
Adversarial Verification are designed to plug into the cascade as
additional ``CascadeStage`` subclasses, not into a logistic
regression.
"""

from .aggregator import ReliabilityAggregator
from .cascade import BeliefStabilityStage, CascadeStage, CascadeStageOutput, ReliabilityCascade, SelfCheckGPTStage
from .cv import document_level_kfold
from .features import (
    DocumentContext,
    SentenceRow,
    attach_graph_scores,
    build_dataset,
    graph_scores_for_document,
)
from .subgroups import compute_subgroup_threshold, filter_subgroup, in_subgroup
from .alpha_selection import select_alpha, auc_pr, recall_at_threshold, ALPHA_GRID, ACTIVE_FEATURES_GRAPH

__all__ = [
    "ReliabilityAggregator",
    "ReliabilityCascade",
    "CascadeStage",
    "CascadeStageOutput",
    "SelfCheckGPTStage",
    "BeliefStabilityStage",
    "document_level_kfold",
    "DocumentContext",
    "SentenceRow",
    "attach_graph_scores",
    "build_dataset",
    "graph_scores_for_document",
    "compute_subgroup_threshold",
    "filter_subgroup",
    "in_subgroup",
    "select_alpha",
    "auc_pr",
    "recall_at_threshold",
    "ALPHA_GRID",
    "ACTIVE_FEATURES_GRAPH",
]
