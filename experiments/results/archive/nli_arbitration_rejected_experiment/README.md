# Rejected experiment: NLI arbitration (Tier 3 matching)

Tested whether enabling `use_nli_arbitration` (DeBERTa-MNLI arbitration
for ambiguous-similarity-band belief matches) improves results, during
the pre-Counterfactual-Verification Belief Stability module review.

- Top-level files here are the **checkpoint** (results before the change).
- `after_nli_on/` holds results **with NLI arbitration enabled**.

**Finding:** standalone Belief Stability AUC-PR improved on all 9
metrics (+0.25 to +1.78pp), but the improvement did not survive contact
with the actual nested-CV Cascade evaluation that matters — the
production `cascade` arm was unchanged within noise (agg 92.68→92.66,
subgroup 87.66→87.99), and the deprecated LR-fused arms saw a mild
regression (−0.13 to −0.32pp aggregate AUC-PR), for ~380x more compute
(0.4s → 155s feature-table build, one GPU model load + inference per
ambiguous pair).

**Decision:** reverted. `use_nli_arbitration` stays `false` in
`configs/default.yaml`. The wiring (`NLIArbitrator` threaded through
`build_dataset`/`evaluate_reliability.py`/`evaluate_belief.py` via
`--use-nli-arbitration`) was kept since it's zero-risk when off and
makes the mechanism testable again in the future.
