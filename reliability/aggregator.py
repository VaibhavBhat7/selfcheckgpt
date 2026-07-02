"""
---------------------------------------------------------
Reliability Aggregation Engine

File        : aggregator.py

Description
-----------
Thin wrapper around sklearn LogisticRegression that takes
NAMED per-sentence signal dicts, not positional feature
vectors.

SUPERSEDED as the production aggregation strategy by the
multi-stage ``ReliabilityCascade`` (see cascade.py) -
experimental comparison (experiments/compare_fusion_strategies.py)
showed a single global LR fit implies a decision boundary on
selfcheck_score far above the 0.5 convention used to evaluate
it, which mechanically under-recalls the consistent-
hallucination subgroup (35.08% recall vs the cascade's 87.38%,
at LOWER aggregate AUC-PR too: 92.21 vs 92.58). Kept here for
the bs_baseline/bs_bayesian/bs_graph comparison arms in
experiments/evaluate_reliability.py and as a documented
alternative - not the recommended path for new signals.
Counterfactual and Adversarial Verification should integrate
as new ``CascadeStage`` subclasses, not as new keys here.
---------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


class ReliabilityAggregator:
    """
    Fuses named confidence signals into one calibrated
    P(non-factual) via logistic regression.
    """

    def __init__(
        self,
        active_features: Sequence[str],
        class_weight: str | None = "balanced",
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:

        self.active_features = list(active_features)

        self.model = LogisticRegression(
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
        )

    # --------------------------------------------------

    def _to_matrix(self, rows: List[Dict[str, float]]) -> np.ndarray:

        return np.array(
            [[row[name] for name in self.active_features] for row in rows],
            dtype=float,
        )

    # --------------------------------------------------

    def fit(
        self,
        rows: List[Dict[str, float]],
        labels: List[int],
    ) -> "ReliabilityAggregator":

        X = self._to_matrix(rows)

        self.model.fit(X, labels)

        return self

    # --------------------------------------------------

    def predict_proba(self, rows: List[Dict[str, float]]) -> np.ndarray:
        """
        Returns P(label=1) i.e. P(non-factual) per row.
        """

        X = self._to_matrix(rows)

        return self.model.predict_proba(X)[:, 1]
