# Evidence-Grounded Verification (EGV)

A hallucination-verification signal that checks a claim against an **external reference document** — Wikipedia's own biography text — instead of checking it against the model's own resampled generations.

Evidence-Grounded Verification is the third stage of the project's **Hybrid Hallucination Detection Framework**, escalation target for the specific population that SelfCheckGPT and Belief Stability both systematically miss: the "consistent-hallucination subgroup." It lives in `reliability/evidence.py` and plugs into the shared `reliability/cascade.py` orchestration engine as one more `CascadeStage`.

---

## Module Overview

EGV takes a claim (a pronoun-resolved sentence from the model's original output) and the corresponding Wikipedia reference passage (`wiki_bio_text`), and runs a genuinely 3-way Natural Language Inference model over the pair to decide whether the evidence **supports**, **contradicts**, or simply **doesn't address** the claim. That third option — abstaining when the evidence is silent — is the module's central design decision (see [Research Idea](#research-idea)) and is what required a different NLI checkpoint from the rest of the project (see [Motivation](#motivation)).

EGV never runs standalone in production; it is the terminal stage of a 3-stage cascade (`SelfCheckGPT -> Belief Stability -> EGV`), consulted only for sentences the first two stages could not confidently resolve.

---

## Motivation

SelfCheckGPT-NLI and Belief Stability are architecturally different (sentence-level entailment vs. structured belief-graph propagation), but they share one deep property: **both are self-referential**. Both ask, in different ways, "does the model's own resampled output agree with itself?" Neither ever consults anything outside the model's own generations.

This project's own diagnostics (see `reliability/subgroups.py`) identified a specific, reproducible blind spot: a subgroup of non-factual sentences where SelfCheckGPT's own consistency score is confidently — and wrongly — low, even at its own ROC-optimal operating threshold. Because Belief Stability's persistence signal is *also* computed from the same resample pool, it shares the same root vulnerability: if the model hallucinates the *same wrong fact* consistently across every resample, no purely self-referential signal — however cleverly structured — can detect it. Consistency with oneself is not the same thing as correctness.

EGV breaks that shared blind spot by consulting a source that never depends on the model's own outputs at all: the ground-truth Wikipedia passage the biography is actually about.

---

## Research Idea

> Two structurally different kinds of hallucination-detection signal exist: **self-referential** (does the model agree with itself?) and **externally-referential** (does an independent source agree with the model?). They are complementary, not redundant — a cascade should escalate specifically to the second kind only once the first kind has been exhausted.

Two research decisions fall directly out of this framing:

1. **Neutral-as-No-Evidence.** A reference document frequently does not address a given claim at all (WikiBio's first paragraph rarely covers everything a generated biography sentence might assert). Forcing a binary support/contradict verdict in that case manufactures a confident-sounding but meaningless signal. EGV instead treats the NLI model's own "neutral" verdict as a legitimate abstention — the stage falls back to whatever the previous cascade stage already decided, rather than overriding it with noise. This requires the entailment model to be able to express neutrality at all (see the checkpoint discussion below).
2. **A recall-oriented pre-filter, not a decision-maker.** Before spending an NLI forward pass, a cheap entity/lexical overlap check (shared proper nouns/numbers between claim and evidence) can rule out claims the evidence clearly doesn't cover — but only to abstain, never to force a verdict; it is deliberately biased toward false "proceed to NLI" calls over false "no evidence" calls.

**A concrete engineering finding shaped the whole design**: the NLI checkpoint SelfCheckGPT and Belief Stability already share (`potsawee/deberta-v3-large-mnli`) is architecturally 2-way — confirmed via its `config.json` (`num_labels=2`) and model card ("neutral class predictions were excluded from the output heads"). There is no neutral signal to recover from it. EGV therefore uses a different, genuinely 3-way checkpoint (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) — same architecture family and scale, additionally trained on FEVER-NLI, the closest published training signal to claim-vs-evidence verification — scoped to EGV **only**. SelfCheckGPT and Belief Stability deliberately keep their original checkpoint unchanged, so their own existing comparison stays clean; EGV's comparison axis is the evidence source itself (resamples vs. an external reference), and a different entailment tool there is a secondary, intentional confound layered on top of that primary one, not a violation of anything.

---

## Architecture / Pipeline

**Offline (once per dataset, GPU):**

```
┌───────────────────────────── OFFLINE (GPU, run once) ──────────────────────────────┐
│                                                                                      │
│   wiki_bio_gpt3_hallucination[i]                                                    │
│     ├─ gpt3_sentences  ──► pronoun resolution (belief_stability.pronoun_resolver)   │
│     │                       identify_primary_subject() + resolve_pronouns()          │
│     │                       (same resolution belief_cache.py applies before REBEL)   │
│     └─ wiki_bio_text  ─────────────────────────────────────────┐                    │
│                                                                  ▼                    │
│                             tokenizer(evidence, claim)  =  (premise, hypothesis)      │
│                             truncation @ 512 tokens (flagged, not silently dropped)   │
│                                                                  │                    │
│                                                                  ▼                    │
│         EGVEntailmentModel  (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)│
│                                                                  │                    │
│                                                                  ▼                    │
│         EGVClaimScore(entailment, neutral, contradiction, truncated)                  │
│                                                                  │                    │
│                                                                  ▼                    │
│                    egv_cache.pkl   {dataset_id: [EGVClaimScore, ...]}                 │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Online (every evaluation run, CPU-only — reads the cache, no model inference):**

```
┌────────────────────────── ONLINE (CPU-only, cache-driven) ─────────────────────────┐
│                                                                                      │
│   flatten_egv_cache()  →  {(dataset_id, sentence_index): EGVClaimScore}             │
│                                        │                                            │
│                                        ▼  EvidenceGroundedVerificationStage.predict()│
│           for each row:                                                             │
│             1. no cached score?                        ──────────► ABSTAIN          │
│             2. entity/lexical pre-filter fails? (use_entity_prefilter)               │
│                zero salient-token overlap w/ evidence   ──────────► ABSTAIN          │
│             3. argmax verdict == "neutral"? (neutral_handling=True, the default)      │
│                                                          ──────────► ABSTAIN          │
│             4. otherwise:                                                            │
│                contradiction_risk = contradiction / (entailment + contradiction)      │
│                blended = blend_weight * incoming_score + (1 - blend_weight) * risk    │
│                                        │                                            │
│                                        ▼                                            │
│           ABSTAIN  →  fall back to the latest score handed down by                   │
│                       ReliabilityCascade (e.g. Belief Stability's blended verdict,    │
│                       not the raw SelfCheckGPT score — see Data Flow)                 │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**How it sits inside the production cascade** (`reliability/cascade.py`):

```
   SelfCheckGPT ──confident (>= ROC-optimal threshold)──► final score
        │ else escalate
        ▼
   Belief Stability (non-terminal here) ──always escalates──► contributes a blended score
        │
        ▼
   Evidence-Grounded Verification (terminal)
        │
        ├─ resolves (evidence available, non-neutral) ──► final score
        └─ abstains ──► forwards Belief Stability's score unchanged (fallback-forwarding)
```

---

## Key Components

| Component | File | Role |
|---|---|---|
| `EGVClaimScore` | `reliability/evidence.py` | Dataclass: `entailment`, `neutral`, `contradiction`, `truncated`; derives `.verdict` (argmax) and `.contradiction_risk` (renormalized over entailment+contradiction). |
| `EvidenceGroundedVerificationStage` | `reliability/evidence.py` | The `CascadeStage` implementation: abstention logic, blending, fallback-score consumption. |
| `extract_salient_tokens` / `passes_entity_prefilter` | `reliability/evidence.py` | Recall-oriented entity/number overlap pre-filter (spaCy `PROPN`/`NUM` tokens). |
| `build_evidence_lookup` | `reliability/evidence.py` | `{dataset_id: wiki_bio_text}` lookup built once per dataset. |
| `flatten_egv_cache` | `reliability/evidence.py` | Converts the per-document cache shape into the flat `(dataset_id, sentence_index)` keying every other stage uses. |
| `EGVEntailmentModel` | `experiments/build_egv_cache.py` | Loads the 3-way checkpoint, tokenizes `(evidence, claim)`, asserts the label space is exactly `{entailment, neutral, contradiction}`. |
| `ReliabilityCascade` / `set_fallback_scores` | `reliability/cascade.py` | Shared orchestration: tracks every stage's latest computed score (not just resolved rows) and offers it to the next stage, so an abstaining EGV falls back to Belief Stability's verdict, not the raw SelfCheckGPT score. |

---

## Directory Structure

```
selfcheckgpt/
├── reliability/
│   ├── evidence.py                      # EGV itself: EGVClaimScore, EvidenceGroundedVerificationStage
│   ├── cascade.py                       # ReliabilityCascade, SelfCheckGPTStage, BeliefStabilityStage (shared)
│   ├── features.py                      # SentenceRow / DocumentContext / build_dataset (shared feature table)
│   ├── subgroups.py                     # consistent-hallucination subgroup definition (shared)
│   ├── alpha_selection.py               # nested-CV alpha selection, auc_pr, recall_at_threshold (shared)
│   ├── cv.py                            # document_level_kfold (shared)
│   ├── aggregator.py                    # ReliabilityAggregator, the superseded LR-fusion baseline (shared)
│   └── __init__.py
├── experiments/
│   ├── build_egv_cache.py               # Offline: 3-way NLI scoring -> egv_cache.pkl
│   ├── diagnose_egv_coverage.py         # Data diagnostic: verdict distribution, truncation risk
│   ├── evaluate_egv.py                 # Steps 5-6: integrate EGV, 6-arm ablation vs. baseline/BS
│   ├── evaluate_ablation_matrix.py      # Cascade combination matrix incl. both order-swaps
│   ├── evaluate_neutral_ablation.py     # Forced-binary ablation isolating neutral-handling's value
│   ├── error_overlap_analysis.py        # Venn-style breakdown: which signal(s) catch which hallucinations
│   ├── pooled_significance_egv.py       # Pooled McNemar's / Sign test (higher power than 5-fold t-test)
│   ├── pooled_significance_order_swap.py# Same, applied to the order-swap comparison
│   ├── bootstrap_confidence_intervals.py# Document-level bootstrap CIs across all arms/comparisons
│   └── results/                         # all generated caches, CSVs, and figures land here
└── tests/
    └── test_cascade.py                  # SelfCheckGPTStage / BeliefStabilityStage / ReliabilityCascade unit tests
```

---

## Important Classes and Files

- **`reliability/evidence.py::EGVClaimScore`** — `entailment: float`, `neutral: float`, `contradiction: float`, `truncated: bool = False`; `.verdict` (argmax label), `.contradiction_risk` (`contradiction / (entailment + contradiction)`, `0.5` if that denominator is `0`).
- **`reliability/evidence.py::EvidenceGroundedVerificationStage`** — constructor: `egv_scores` (flattened cache), `evidence_lookup=None`, `use_entity_prefilter=True`, `blend_weight=0.3`, `is_terminal=True`, `neutral_handling=True`. Implements `set_fallback_scores()` so it can consume whatever the previous cascade stage computed.
- **`reliability/cascade.py::ReliabilityCascade`** — orchestrates an ordered list of stages; the fallback-forwarding mechanism that lets EGV's abstention forward Belief Stability's score (not raw SelfCheckGPT) lives here.
- **`experiments/build_egv_cache.py::EGVEntailmentModel`** — `MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"`, `MAX_LENGTH = 512`; asserts `set(label_index) == {"entailment", "neutral", "contradiction"}` at construction time (fails loudly if pointed at an incompatible checkpoint).
- **`experiments/evaluate_ablation_matrix.py`** — the most complete single ablation table: `selfcheckgpt`, `selfcheckgpt_bs`, `selfcheckgpt_egv`, `bs_egv`, `cascade_full`, `egv_bs`, `cascade_egv_first`.

---

## Data Flow

```
wiki_bio_gpt3_hallucination[i]
  ├─ gpt3_sentences  ──► pronoun-resolved  ──► claim (one per sentence)
  └─ wiki_bio_text   ─────────────────────────────────────► evidence
        │
        ▼  EGVEntailmentModel.score(evidence, claim)   [offline, once]
  EGVClaimScore  ──►  egv_cache.pkl  {dataset_id: [EGVClaimScore, ...]}
        │
        ▼  flatten_egv_cache()   [online, every run]
  {(dataset_id, sentence_index): EGVClaimScore}
        │
        ▼  EvidenceGroundedVerificationStage.predict(rows)
        │     incoming = fallback_scores.get(key, row.selfcheck_score)   # set by ReliabilityCascade
        ▼
  CascadeStageOutput(score, resolved, stage="egv")
```

`reliability/features.py::build_dataset` is what populates `row.claim_text` with the pronoun-resolved sentence (reusing `belief_stability.pronoun_resolver`, not REBEL) — EGV needs only the resolved sentence text, not Belief Stability's structured triples.

---

## Configuration Options

`EvidenceGroundedVerificationStage`'s constructor is where EGV's behavior is tuned (there is no separate YAML section for it in `configs/default.yaml` — its one config-file-shared value is `cascade_blend_weight`, which `BeliefStabilityStage` uses; EGV's own blend weight is passed as a CLI flag, see below):

| Option | Default | Effect |
|---|---|---|
| `use_entity_prefilter` | `True` | Abstain if the claim's salient tokens (proper nouns/numbers) have zero overlap with the evidence text. |
| `blend_weight` | `0.3` | Weight on the incoming score vs. `(1 - blend_weight)` on EGV's own `contradiction_risk`. |
| `is_terminal` | `True` | Whether this stage resolves a row outright or escalates it further. |
| `neutral_handling` | `True` | When `False`, disables *only* the neutral-verdict abstention — every cached, prefilter-passing claim is forced through the entailment/contradiction blend regardless of verdict. Added specifically to support the forced-binary ablation (see [Current Results](#current-results--key-findings)); default preserves all other behavior exactly. |

CLI-level knob across every `experiments/*.py` script: `--egv-blend-weight` (default `0.3`).

---

## Dependencies

Same core stack as the rest of the project, plus nothing EGV-specific beyond what's already required:

```
torch
transformers
datasets
spacy               # + en_core_web_sm (used by the entity/lexical pre-filter and pronoun resolution)
scikit-learn        # CV splitting, precision_recall_curve, roc_curve
scipy               # binomtest (McNemar's/Sign tests), ttest_rel, beta (transitively, via Belief Stability)
PyYAML              # BeliefStabilityConfig, loaded by every evaluate_*.py script
matplotlib
pytest
```

EGV's own model, pulled automatically from the HuggingFace Hub on first use: [`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli). Every `evaluate_*.py` script in this module also builds the *full* feature table (`reliability/features.py::build_dataset`), so it transitively needs `belief_cache.pkl` and `selfcheck_cache.pkl` too — see [How to Run](#how-to-run).

---

## How to Run

### 1. Set up the environment

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux

pip install torch transformers datasets spacy scikit-learn scipy PyYAML matplotlib pytest sentence-transformers
python -m spacy download en_core_web_sm
```

### 2. Build every prerequisite cache (run once each)

EGV's evaluation scripts build the full 3-signal feature table, so all three caches are required — none are shipped in the repository (`experiments/results/*.pkl` is gitignored):

```bash
python experiments/build_belief_cache.py --resume          # REBEL claims  -> claim_cache.pkl
python experiments/canonicalize_cache.py                    # canonicalize -> belief_cache.pkl (+ .embeddings.pkl)
python experiments/build_selfcheck_cache.py --resume        # SelfCheckNLI  -> selfcheck_cache.pkl
python experiments/build_egv_cache.py --limit 5             # EGV smoke test first
python experiments/build_egv_cache.py --resume               # EGV, full dataset -> egv_cache.pkl
```

### 3. Run the module independently (data diagnostic — no fusion)

```bash
python experiments/diagnose_egv_coverage.py
```

Reports the verdict distribution (support/neutral/contradiction) and truncation rate directly from `egv_cache.pkl`, with no dependence on the belief/selfcheck caches.

### 4. Run the evaluation (fused with SelfCheckGPT + Belief Stability)

```bash
python experiments/evaluate_egv.py
```

### 5. Reproduce every reported ablation / significance result

```bash
python experiments/evaluate_ablation_matrix.py            # full combination matrix incl. both order-swaps
python experiments/evaluate_neutral_ablation.py            # forced-binary ablation
python experiments/error_overlap_analysis.py                # error-overlap breakdown
python experiments/pooled_significance_egv.py                # pooled McNemar's / Sign tests
python experiments/pooled_significance_order_swap.py          # pooled tests for the order-swap comparison
python experiments/bootstrap_confidence_intervals.py          # bootstrap CIs across all arms/comparisons
```

All six accept `--egv-blend-weight`, `--outer-k`, `--inner-k`, and `--seed` if you want to deviate from the reported defaults (`0.3`, `5`, `3`, `42`).

### 6. Run the tests

```bash
python -m pytest tests/ -q
```

---

## Expected Outputs

| Script | Primary output(s) |
|---|---|
| `build_egv_cache.py` | `experiments/results/egv_cache.pkl` |
| `diagnose_egv_coverage.py` | console report only (no file output) |
| `evaluate_egv.py` | `egv_cv_results.csv`, `egv_summary.csv`, `significance_tests_egv.csv`, `pr_curve_egv_arms.png`, `subgroup_recall_egv.png` |
| `evaluate_ablation_matrix.py` | `ablation_matrix_cv_results.csv`, `ablation_matrix_summary.csv`, `significance_tests_ablation_matrix.csv`, `pr_curve_ablation_matrix.png`, `subgroup_recall_ablation_matrix.png` |
| `evaluate_neutral_ablation.py` | `neutral_ablation_cv_results.csv`, `neutral_ablation_summary.csv`, `significance_tests_neutral_ablation.csv`, `pr_curve_neutral_ablation.png`, `subgroup_recall_neutral_ablation.png` |
| `error_overlap_analysis.py` | `error_overlap_analysis.csv`, `error_overlap_analysis.png` |
| `pooled_significance_egv.py` | `pooled_significance_egv.csv` |
| `pooled_significance_order_swap.py` | `pooled_significance_order_swap.csv` |
| `bootstrap_confidence_intervals.py` | `bootstrap_arm_cis.csv`, `bootstrap_paired_diffs.csv` |

## Generated CSVs/Figures

- **`egv_summary.csv` / `ablation_matrix_summary.csv` / `neutral_ablation_summary.csv`** — columns `arm, agg_mean, agg_std, sub_recall_mean, sub_recall_std, n_folds`, one row per arm.
- **`significance_tests_*.csv`** — paired t-tests across the 5 outer folds: `arm_a, arm_b, metric, mean_diff, ci95_low, ci95_high, t_stat, p_value, n_folds`.
- **`pooled_significance_egv.csv` / `pooled_significance_order_swap.csv`** — row-level McNemar's exact + Sign tests on pooled out-of-fold subgroup predictions: `arm_a, arm_b, test, n_pooled, recall_a, recall_b, count_a_better, count_b_better, count_tied_or_neither, n_effective, p_value, significant_p05`.
- **`bootstrap_arm_cis.csv`** — per-arm point estimate + 95% CI for both metrics: `arm, agg_point, agg_ci_low, agg_ci_high, sub_point, sub_ci_low, sub_ci_high`.
- **`bootstrap_paired_diffs.csv`** — paired bootstrap CIs on the difference between two arms: `arm_a, arm_b, metric, observed_diff, ci_low, ci_high, p_boot_approx, significant_ci_excludes_0`.
- **`error_overlap_analysis.csv`** — `population, category, count, n_total, pct`, where `category` is one of `all_three, sc_bs_only, sc_egv_only, bs_egv_only, sc_only, bs_only, egv_only, missed_by_all`.

---

## How it integrates with the overall SelfCheckGPT framework

EGV is the third and terminal stage of the production `ReliabilityCascade`:

```python
ReliabilityCascade([
    SelfCheckGPTStage(threshold=threshold),
    BeliefStabilityStage(contexts=contexts, alpha=alpha_star, ..., is_terminal=False),
    EvidenceGroundedVerificationStage(egv_scores=egv_scores, evidence_lookup=evidence_lookup,
                                       use_entity_prefilter=True, blend_weight=0.3, is_terminal=True),
])
```

Rows SelfCheckGPT is already confident about (score at or above its ROC-optimal threshold) never reach EGV at all. Rows Belief Stability escalates arrive at EGV carrying Belief Stability's own blended score as their "incoming" value (via `ReliabilityCascade`'s fallback-forwarding — every stage's computed score, not just resolved rows' scores, is tracked and offered to the next stage). If EGV abstains (no evidence, prefilter fails, or neutral verdict), that Belief Stability score passes through unchanged rather than being silently discarded — this was a real architectural bug, discovered and fixed during this project's own evaluation discipline (see the git history / `reliability/cascade.py`'s docstring for the full account).

---

## Current Results / Key Findings

From the 5-fold, document-level nested-CV protocol shared across every script in this module:

| Arm | Aggregate AUC-PR | Subgroup Recall@0.5 |
|---|---|---|
| `selfcheckgpt` (baseline) | 92.62 ± 0.92 | 67.99 ± 3.61 |
| `selfcheckgpt_bs` (2-stage cascade) | 92.68 ± 0.95 | 87.66 ± 4.46 |
| **`cascade_full`** (3-stage, production) | **92.79 ± 0.88** | **89.42 ± 2.82** |

**Headline claim, confirmed by three independent methods** (paired t-test, pooled McNemar's/Sign test, document-level bootstrap): `cascade_full` beats raw SelfCheckGPT decisively on the subgroup it exists to fix (+21 points, `p < 0.001` under every method tried).

**EGV's marginal contribution beyond Belief Stability is real but modest, and the two available tests answer genuinely different questions** — the honest, non-overclaiming summary: EGV provides a systematic, positively-directed correction to Belief Stability's score (Sign test on raw score direction, `p < 0.0001`: EGV moves the risk score in the correct direction roughly 4–5x more often than the wrong one, on the ~30% of subgroup rows where it doesn't abstain), but that correction is not yet large enough to flip a significant number of final binary decisions at the standard 0.5 threshold (McNemar's on binarized correctness, bootstrap CI on the recall difference: both not significant, `p ≈ 0.18–0.24`).

**Neutral-handling is load-bearing** — removing it (forcing every cached claim through a binary support/contradict call, ignoring the model's own "neutral" verdict) collapses `cascade_full`'s subgroup recall from 89.42% to 70.97% (`p = 0.0024`), because an abstaining EGV's real job inside the cascade is protecting Belief Stability's already-good score from being overwritten by a forced, often-spurious call — not merely "avoiding noise" in isolation.

**Ordering the two escalation stages the other way (EGV before Belief Stability) does not materially change the framework**: both orderings land at statistically indistinguishable final recall (~89–90%), though the current ordering does produce systematically better-directed underlying scores (Sign test `p < 0.0001`) — a real but practically modest reason to prefer the shipped design, not evidence the alternative would perform worse.

**Error-overlap**: of all non-factual sentences, 92.0% are caught by all three signals (SelfCheckGPT, Belief Stability, EGV) simultaneously; restricted to the hard subgroup, Belief Stability's unique contribution (16.6% of the subgroup) is roughly 8x larger than EGV's unique contribution (2.2%); 8.9% of the subgroup is missed by all three signals — the framework's honestly-quantified remaining blind spot.

Full detail for every claim above is in `experiments/results/*.csv`.

---

## Current Limitations

- **Whole-document evidence, no claim-level localization.** EGV checks a claim against the *entire* `wiki_bio_text` passage rather than a retrieved, claim-relevant span — deferred by design (see Future Improvements) to keep the first implementation simple and dataset-independent-in-principle.
- **Truncation risk**: ~11% of (evidence, claim) pairs exceed the checkpoint's 512-token limit and get silently truncated (flagged via `EGVClaimScore.truncated`, but not otherwise corrected for).
- **Evidence silence is common**: roughly half of all claims are evidence-silent (neutral verdict) on WikiBio's short reference passages — this bounds how much of the subgroup EGV can ever address, independent of model quality.
- **A confound is baked into the design on purpose**: EGV's checkpoint differs from SelfCheckGPT/Belief Stability's not only in training data (FEVER-NLI) but also in being genuinely 3-way vs. 2-way — EGV's "vs. self-referential signals" comparison is clean on the evidence-source axis, but not a controlled single-variable comparison against the shared checkpoint.
- **Reference evidence is treated as ground truth.** `wiki_bio_text` (a Wikipedia biography's first paragraph) can itself be incomplete, outdated, or wrong; EGV has no notion of evidence reliability or evidence quality.
- **Blend weights and thresholds are hand-set, not learned or calibrated** (`blend_weight=0.3`, the SelfCheckGPT gate's ROC-optimal threshold, Belief Stability's `alpha`) — the cascade is a validated engineering design, not a formally derived probabilistic fusion.
- **Single-dataset validation.** Every result above is on `potsawee/wiki_bio_gpt3_hallucination`; the "external evidence" concept generalizes conceptually, but nothing here has been tested against a dataset where reference evidence is retrieved rather than given.

---

## Future Improvements

- **Claim-level evidence localization / retrieval** (rather than whole-document evidence), extending naturally toward multi-hop, multi-fragment evidence reasoning instead of a single NLI call per claim.
- **Evidence-reliability modeling** — represent uncertainty *in the evidence itself* (completeness, staleness) rather than treating the reference document as certain ground truth.
- **A principled probabilistic fusion layer** replacing the hand-set blend weights/thresholds throughout the cascade, so the combination rule is derived rather than validated post hoc.
- **Learned, instance-adaptive routing** between stages, motivated by a formal taxonomy of verification-signal types (self-referential vs. externally-referential vs. a hypothetical third axis grounded in model-internal signals), instead of a fixed stage order.
- **Cross-dataset validation**, once evidence retrieval (rather than a given reference passage) makes the framework meaningfully dataset-independent.

---

## References

- Model checkpoint (this module's core dependency): [`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli).
- He, P., Gao, J., & Chen, W. (2021/2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.*
- Williams, A., Nangia, N., & Bowman, S. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference* (MultiNLI).
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification.*
- Nie, Y., Williams, A., Dinan, E., Bansal, M., Weston, J., & Kiela, D. (2020). *Adversarial NLI: A New Benchmark for Natural Language Understanding* (ANLI).
- Liu, A., et al. (2022). *WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation.*
- Manakul, P., Liusie, A., & Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
- Dataset: [`potsawee/wiki_bio_gpt3_hallucination`](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination).

---

*EGV is the third stage of the project's Hybrid Hallucination Detection Framework. See `README_BELIEF_STABILITY.md` for the second stage, and `reliability/cascade.py` for the shared orchestration engine both plug into.*
