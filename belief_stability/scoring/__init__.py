"""
Belief Scoring Package.
"""

from .persistence import BeliefPersistence
from .base import BaseBeliefScorer, aggregate
from .baseline_scorer import BaselineScorer, BeliefScorer
from .bayesian_scorer import BayesianScorer
from .graph_scorer import GraphScorer

_SCORERS = {
    "baseline": BaselineScorer,
    "bayesian": BayesianScorer,
    "graph": GraphScorer,
}


def build_scorer(
    method: str = "bayesian",
    absent_discount: float = 0.5,
    graph_alpha: float = 0.3,
) -> BaseBeliefScorer:
    """
    Factory selecting a scoring strategy by name.

    Parameters
    ----------
    method : str
        One of "baseline", "bayesian", "graph".
    """

    if method not in _SCORERS:

        raise ValueError(
            f"Unknown scoring method '{method}'. "
            f"Expected one of {sorted(_SCORERS)}."
        )

    if method == "baseline":
        return BaselineScorer()

    if method == "bayesian":
        return BayesianScorer(absent_discount=absent_discount)

    return GraphScorer(alpha=graph_alpha, absent_discount=absent_discount)


__all__ = [
    "BeliefPersistence",
    "BaseBeliefScorer",
    "aggregate",
    "BaselineScorer",
    "BeliefScorer",
    "BayesianScorer",
    "GraphScorer",
    "build_scorer",
]
