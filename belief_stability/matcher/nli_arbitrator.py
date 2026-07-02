"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Belief Matching

File        : nli_arbitrator.py

Description
-----------
Optional Tier-3 matcher. Reuses the same DeBERTa-v3-large
MNLI checkpoint SelfCheckGPT-NLI already relies on
(potsawee/deberta-v3-large-mnli) to arbitrate the narrow
band of belief pairs where semantic similarity is
ambiguous (neither a confident SUPPORT nor a confident
CONTRADICT).

This tier is disabled by default (see
BeliefStabilityConfig.use_nli_arbitration) because it is
the only tier that runs a transformer online. It is
implemented so it can be switched on for an ablation run
without reintroducing GPU calls into the default path.
---------------------------------------------------------
"""

from __future__ import annotations

import torch

from belief_stability.models import Belief


class NLIArbitrator:

    MODEL_NAME = "potsawee/deberta-v3-large-mnli"

    def __init__(
        self,
        device: str | None = None,
    ) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            use_fast=False,
        )

        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        )

        self.model.eval()

        self.model.to(self.device)

    # --------------------------------------------------

    @staticmethod
    def _render(belief: Belief) -> str:

        return f"{belief.subject} {belief.relation.replace('_', ' ').lower()} {belief.object}"

    # --------------------------------------------------

    @torch.no_grad()
    def contradiction_probability(
        self,
        original_belief: Belief,
        candidate_belief: Belief,
    ) -> float:
        """
        P(contradiction) between the candidate belief
        (premise) and the original belief (hypothesis),
        following the same convention as SelfCheckNLI.
        """

        premise = self._render(candidate_belief)

        hypothesis = self._render(original_belief)

        inputs = self.tokenizer(
            hypothesis,
            premise,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)

        # index 1 == contradiction, following SelfCheckNLI's convention
        # (neutral already collapsed out of this checkpoint's label space).
        return float(probs[0][1].item())
