# Belief Stability

A structured, graph-based hallucination-detection signal that checks whether the *factual claims* embedded in an LLM's output remain **persistently supported** across the model's own resampled generations — as opposed to checking surface-level textual consistency.

Belief Stability is a component of a larger **Hybrid Hallucination Detection Framework** built on top of [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt). It is designed as a self-contained Python package (`belief_stability/`) with its own offline caching pipeline, its own configuration surface, and its own standalone evaluation harness, while also plugging into the framework's shared `reliability/` fusion engine as one stage of a multi-stage detection cascade.

`__version__ = "2.0.0"` (see `belief_stability/__init__.py`).

---

## Module Overview

Given an LLM-generated passage and several stochastically resampled alternative generations of the same passage (the standard SelfCheckGPT setup), Belief Stability:

1. **Extracts structured claims** — `(subject, relation, object)` triples — from the original passage and from every resample, using relation extraction (REBEL) rather than treating each sentence as an opaque string.
2. **Canonicalizes** those claims into a normalized vocabulary so that semantically identical facts expressed with different surface wording (different relation phrasing, entity aliases, pronouns, nicknames) collapse to the same *belief*.
3. **Matches** each belief extracted from the original passage against the beliefs extracted from every resample, via a tiered cascade (exact match → inverse/symmetric relation match → semantic embedding match → optional NLI arbitration), producing a `SUPPORT` / `ABSENT` / `CONTRADICT` verdict per (belief, resample) pair.
4. **Persists** those verdicts into a per-belief profile (counts and confidence-weighted sums of support/absent/contradict).
5. **Scores** each profile into a stability value in `[-1, 1]` (`+1` = consistently supported, `-1` = consistently contradicted), using one of three interchangeable scoring strategies (`baseline`, `bayesian`, or the flagship `graph` scorer, which propagates consistency information across beliefs that share an entity).

The result is a per-sentence signal that is fused with SelfCheckGPT's own resampling-consistency score inside the project's `ReliabilityCascade` (see [How it integrates with the overall SelfCheckGPT framework](#how-it-integrates-with-the-overall-selfcheckgpt-framework)).

---

## Motivation

SelfCheckGPT's NLI variant detects hallucination by checking whether a sentence is textually entailed by the model's own resampled generations. This is a strong, general-purpose signal, but it is **self-referential**: it only asks "does the model say something similar again?", not "is this specific fact — this subject/relation/object — actually the same fact across resamples?"

Two sentences can be judged "similar enough" by a sentence-level NLI model while disagreeing about the specific fact that matters (e.g. a birth year, a job title, a spouse's name), and conversely two sentences can look superficially different (paraphrase, pronoun substitution, reordering) while asserting exactly the same underlying fact. SelfCheckGPT has a measurable, empirically diagnosed blind spot: a subgroup of non-factual sentences that its own resampling-consistency score confidently — and wrongly — scores as *low risk* (see `reliability/subgroups.py`'s "consistent-hallucination subgroup").

Belief Stability addresses this by re-framing the consistency question at the level of **structured facts** instead of **raw sentences**: extract what is actually being claimed, normalize it, and check *that specific claim's* persistence across resamples, using a belief graph rather than a single pairwise text comparison.

---

## Research Idea

> A hallucinated fact is not merely a sentence the model happens to phrase differently across samples — it is a *specific claim* (subject, relation, object) that fails to recur, or is actively contradicted, once you strip away surface-level phrasing differences and look at the underlying entity/relation structure.

This reframes hallucination detection as a **claim persistence** problem:

- Represent each sentence as one or more discrete beliefs (structured triples), not a single embedding or entailment score.
- Match beliefs across resamples on the *content* of the claim, tolerant to paraphrase, entity aliasing, and pronoun reference — but strict about the underlying relation.
- Aggregate per-belief evidence into a **belief graph**, where beliefs sharing an entity (e.g. two different claims both about "Marie Curie") can propagate consistency information to each other — a structural signal SelfCheckGPT's per-sentence scoring has no analogue for.
- Treat "the resample never mentions this claim at all" (`ABSENT`) as a distinct, softer signal from an explicit `CONTRADICT`, rather than conflating both into "not supported."

---

## Architecture / Pipeline

Two parallel tracks: an **offline / GPU** track (claim extraction — expensive, run once) and an **offline-then-online / CPU-only** track (canonicalization, matching, scoring — cheap, safe to re-run for every ablation).

```
┌─────────────────────────────── OFFLINE (GPU, run once) ────────────────────────────────┐
│                                                                                          │
│   Original passage + N sampled passages (wiki_bio_gpt3_hallucination)                  │
│                              │                                                          │
│                              ▼   RebelClaimExtractor  (Babelscape/rebel-large)          │
│                     ExtractedClaim triples: (subject, relation, object)                 │
│                              │                                                          │
│                              ▼   RawClaimCacheBuilder.build()                           │
│                        claim_cache.pkl   (ClaimCache: raw, uncanonicalized)             │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼─────────────── OFFLINE (CPU-only, fast) ────────────────┐
│                              ▼                                                          │
│                     CacheCanonicalizer.canonicalize_cache()                             │
│                              │                                                          │
│         Canonicalizer:  Preprocessor → RelationMapper → EntityMapper →                  │
│                          AttributeMapper → BeliefBuilder                                │
│                              │                                                          │
│         + DocumentEntityNormalizer (document-scoped alias merging: titles,              │
│           surname→full-name, nickname→full-name — all gated by an ambiguity check)      │
│                              │                                                          │
│                              ▼                                                          │
│        belief_cache.pkl (BeliefCache: canonical Belief objects)                         │
│        belief_cache.pkl.embeddings.pkl (SemanticMatcher embedding cache)                │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼─────────────── ONLINE (CPU-only, every eval run) ───────┐
│                              ▼                                                          │
│              BeliefStabilityPipeline.run_from_beliefs()                                 │
│                              │                                                          │
│                              ▼   BeliefMatcher.match_all()                              │
│              BeliefLookup (index by (subject, relation))                                │
│                              │                                                          │
│                              ▼   TransitionClassifier.classify()   — tier cascade:       │
│              Tier 1  EXACT     candidate.object == original.object                      │
│              Tier ●  INVERSE   structurally-verified inverse/symmetric relation match    │
│              Tier 2  SEMANTIC  cosine similarity(candidate.object, original.object)      │
│                                >= match_similarity_threshold  (needs SemanticMatcher)    │
│              Tier 3  NLI       only in the ambiguous similarity band; off by default     │
│                                (needs NLIArbitrator + use_nli_arbitration=true)          │
│              fallback          CONTRADICT (candidates exist but none matched)            │
│                              │                                                          │
│                              ▼   TransitionResult  (SUPPORT / ABSENT / CONTRADICT)       │
│                              │                                                          │
│                              ▼   BeliefPersistence.compute()                            │
│              BeliefProfile  (support / absent / contradict counts + confidence weights)  │
│                              │                                                          │
│                              ▼   BaselineScorer | BayesianScorer | GraphScorer           │
│              BeliefStabilityResult.stability_score  ∈ [-1, 1]                           │
│              (+1 = consistently supported, -1 = consistently contradicted)              │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

| Component | File | Role |
|---|---|---|
| `RebelClaimExtractor` | `claim_extractor.py` | REBEL-based relation extraction: passage text → `ExtractedClaim` triples. |
| `RawClaimCacheBuilder` / `ClaimCache` | `belief_cache.py` | Batches extraction across a whole dataset; persists raw, uncanonicalized claims. |
| `Canonicalizer` | `canonicalizer/canonicalizer.py` | 5-step pipeline turning one raw `ExtractedClaim` into one canonical `Belief`. |
| `DocumentEntityNormalizer` | `canonicalizer/document_entity_normalizer.py` | Document-scoped entity alias merging (titles, surnames, nicknames), gated by an ambiguity check. |
| `CacheCanonicalizer` | `canonicalize_cache.py` | Applies `Canonicalizer` + `DocumentEntityNormalizer` across a whole `ClaimCache`; builds the semantic-embedding cache. |
| `BeliefMatcher` / `BeliefLookup` | `matcher/matcher.py`, `matcher/lookup.py` | Index-building and candidate retrieval (direct + inverse) for one original belief against one sampled passage. |
| `TransitionClassifier` | `matcher/transition_classifier.py` | The actual tier-cascade decision logic (EXACT → INVERSE → SEMANTIC → NLI → fallback). |
| `SemanticMatcher` | `matcher/semantic.py` | Tier-2 paraphrase matching via cached sentence embeddings. |
| `NLIArbitrator` | `matcher/nli_arbitrator.py` | Tier-3, disabled-by-default NLI tie-breaker for ambiguous similarity scores. |
| `BeliefPersistence` | `scoring/persistence.py` | Aggregates per-(belief, resample) transition results into one `BeliefProfile` per belief. |
| `BaselineScorer` / `BayesianScorer` / `GraphScorer` | `scoring/*.py` | Three interchangeable ways to turn a `BeliefProfile` into a stability score. |
| `BeliefStabilityPipeline` | `pipeline.py` | End-to-end orchestrator; GPU-free once beliefs are cached. |
| `SentenceBeliefRunner` | `sentence_runner.py` | Adapter exposing the pipeline as a SelfCheckGPT-style `predict()`-shaped callable. |

---

## Directory Structure

```
selfcheckgpt/
├── belief_stability/
│   ├── __init__.py                      # public API surface
│   ├── config.py                        # BeliefStabilityConfig
│   ├── models.py                        # Belief, BeliefProfile, Transition, MatchTier, ...
│   ├── constants.py                     # RELATION_MAPPING, INVERSE_RELATION_PAIRS, ENTITY_ALIASES, ...
│   ├── nlp.py                           # shared spaCy (en_core_web_sm) singleton loader
│   ├── sentence_splitter.py             # split_into_sentences()
│   ├── pronoun_resolver.py              # identify_primary_subject(), resolve_pronouns()
│   ├── claim_extractor.py               # ClaimExtractor (ABC), RebelClaimExtractor
│   ├── belief_cache.py                  # ClaimCache/ExampleClaims (raw) + BeliefCache/ExampleBeliefs (canonical)
│   ├── canonicalize_cache.py            # CacheCanonicalizer (cache-level canonicalization)
│   ├── canonicalizer/
│   │   ├── canonicalizer.py             # Canonicalizer (per-claim, 5-step orchestrator)
│   │   ├── preprocessing.py             # Preprocessor (syntactic cleanup only)
│   │   ├── relation_mapper.py           # RelationMapper
│   │   ├── entity_mapper.py             # EntityMapper (global aliases: "USA" -> "United States")
│   │   ├── attribute_mapper.py          # AttributeMapper
│   │   ├── builder.py                   # BeliefBuilder (assigns belief_id)
│   │   └── document_entity_normalizer.py
│   ├── matcher/
│   │   ├── matcher.py                   # BeliefMatcher (retrieval orchestration)
│   │   ├── lookup.py                    # BeliefLookup (index + inverse lookup)
│   │   ├── transition_classifier.py     # TransitionClassifier (the tier cascade)
│   │   ├── semantic.py                  # SemanticMatcher (Tier 2)
│   │   └── nli_arbitrator.py            # NLIArbitrator (Tier 3)
│   ├── scoring/
│   │   ├── base.py                      # BaseBeliefScorer (ABC), aggregate()
│   │   ├── persistence.py               # BeliefPersistence
│   │   ├── baseline_scorer.py           # BaselineScorer (+ BeliefScorer alias)
│   │   ├── bayesian_scorer.py           # BayesianScorer
│   │   └── graph_scorer.py              # GraphScorer
│   ├── pipeline.py                      # BeliefStabilityPipeline
│   ├── sentence_runner.py               # SentenceBeliefRunner
│   ├── evaluation.py                    # LABEL_MAPPING + AUC-PR helpers vs. human labels
│   └── utils.py                         # setup_logger, pretty_print, json/dir helpers
├── configs/
│   └── default.yaml                     # default BeliefStabilityConfig values
├── experiments/
│   ├── build_belief_cache.py            # Stage 1 (GPU): REBEL extraction -> claim_cache.pkl
│   ├── canonicalize_cache.py            # Stage 2 (CPU): claim_cache.pkl -> belief_cache.pkl + .embeddings.pkl
│   ├── evaluate_belief.py               # Stage 3 (CPU): belief_cache.pkl -> AUC-PR ablation table + PR curves
│   ├── measure_match_tiers.py           # diagnostic: tier/transition rate breakdown
│   ├── relation_frequency_report.py     # diagnostic: RELATION_MAPPING coverage report
│   ├── compare_versions.py              # diagnostic: diff two belief_cache.pkl versions
│   └── results/                         # all generated caches, CSVs, and figures land here
└── tests/
    ├── test_belief_cache.py
    ├── test_canonicalizer.py
    ├── test_canonicalize_cache.py
    ├── test_matcher.py
    ├── test_inverse_matching.py
    ├── test_document_entity_normalizer.py
    ├── test_pronoun_resolver.py
    └── test_scoring.py
```

---

## Important Classes and Files

- **`belief_stability/models.py`** — every dataclass in the module: `ExtractedClaim` (raw), `Belief` (canonical, `slots=True`), `PassageBeliefs`, `Transition` (`SUPPORT`/`ABSENT`/`CONTRADICT` enum), `MatchTier` (`EXACT`/`INVERSE`/`SEMANTIC`/`NLI`/`NONE` enum), `TransitionResult`, `BeliefProfile`, `BeliefStabilityResult`.
- **`belief_stability/config.py`** — `BeliefStabilityConfig`, a single flat dataclass (deliberately plain YAML, not Hydra — "a single researcher's project, not a multi-node sweep"), with `from_yaml()`/`to_yaml()`.
- **`belief_stability/pipeline.py`** — `BeliefStabilityPipeline`, with two public entry points: `run()` (raw-claims path, does canonicalization) and `run_from_beliefs()` (cached path, GPU-free — what all batch evaluation uses).
- **`belief_stability/matcher/transition_classifier.py`** — this is where the tier cascade is actually decided (`matcher/matcher.py` only does retrieval/indexing and delegates here).
- **`belief_stability/scoring/graph_scorer.py`** — `GraphScorer`, the flagship scoring arm: builds one belief-graph per passage (beliefs sharing an entity string are connected), computes a Bayesian posterior score per belief, then propagates `(1 - alpha) * own_score + alpha * mean(neighbor_scores)`.
- **`experiments/build_belief_cache.py`** / **`experiments/canonicalize_cache.py`** / **`experiments/evaluate_belief.py`** — the three-stage offline→online CLI pipeline (see [How to Run](#how-to-run)).

---

## Data Flow

```
wiki_bio_gpt3_hallucination[i]
  ├─ wiki_bio_test_idx  ───────────────────────────────► dataset_id (cache key)
  ├─ gpt3_sentences (original, pre-split)  ───► RebelClaimExtractor ───► original_claims: List[List[ExtractedClaim]]
  └─ gpt3_text_samples (resamples, raw text)
        │  split_into_sentences() per sample
        ▼
     RebelClaimExtractor  ───► sampled_claims: List[List[ExtractedClaim]]  (one flat list per sample)
        │
        ▼  ClaimCache entry: ExampleClaims(dataset_id, primary_subject, original_claims, sampled_claims)
        │
        ▼  CacheCanonicalizer.canonicalize_example()
     Canonicalizer (per claim)  +  DocumentEntityNormalizer (per document, both original & sampled pooled together)
        │
        ▼  BeliefCache entry: ExampleBeliefs(dataset_id, original_beliefs, sampled_beliefs)
        │
        ▼  BeliefMatcher.match_all(original_passage, each sampled_passage)
     List[TransitionResult]  (one per (original belief, sampled passage) pair)
        │
        ▼  BeliefPersistence.compute()
     List[BeliefProfile]  (one per distinct original belief, aggregated across ALL samples)
        │
        ▼  <scorer>.compute()
     BeliefStabilityResult(stability_score, profiles, method)
```

Downstream, `reliability/features.py::build_dataset` groups `BeliefProfile`s back into **per-sentence** rows (via `Belief.belief_id -> sentence_index`) and sums `support`/`absent`/`contradict` across every belief in that sentence, matching is performed **once per document** (not per sentence) so `GraphScorer` can see entity-sharing edges across different sentences of the same passage.

---

## Configuration Options

All fields live in `belief_stability/config.py::BeliefStabilityConfig`, with defaults mirrored exactly in `configs/default.yaml`:

| Field | Default | Meaning |
|---|---|---|
| `extractor_batch_size` | `16` | REBEL batch size. |
| `num_beams` | `1` | REBEL generation beam width. |
| `precision` | `"fp16"` | REBEL inference precision (fp16 only takes effect on CUDA). |
| `max_input_length` / `max_output_length` | `512` / `512` | REBEL tokenizer truncation limits. |
| `flush_every` | `10` | Documents processed between cache checkpoints. |
| `use_pronoun_resolution` | `true` | Resolve 3rd-person singular pronouns to the document's primary subject before extraction. |
| `use_document_entity_normalization` | `true` | Ablation toggle for `DocumentEntityNormalizer`. |
| `use_inverse_matching` | `true` | Ablation toggle for the INVERSE match tier. |
| `use_semantic_matching` | `true` | Enables Tier 2 — **also requires a `SemanticMatcher` instance to be passed in by the caller** (see [Current Limitations](#current-limitations)). |
| `match_similarity_threshold` | `0.82` | Cosine similarity cutoff for a Tier-2 SUPPORT. |
| `use_nli_arbitration` | `false` | Enables Tier 3 — **also requires an `NLIArbitrator` instance to be passed in**. |
| `nli_ambiguous_low` / `nli_ambiguous_high` | `0.55` / `0.82` | Similarity band in which Tier 3 is consulted. |
| `scoring_method` | `"bayesian"` | `"baseline"` \| `"bayesian"` \| `"graph"`. |
| `absent_discount` | `0.5` | How much an `ABSENT` transition counts as evidence of instability, relative to a full `CONTRADICT` (Bayesian/Graph scorers). |
| `graph_alpha` | `0.3` | Neighbor-propagation weight in `GraphScorer`. |
| `sentence_aggregation` | `"min"` | `"min"` \| `"mean"` — how multiple beliefs in one sentence are re-aggregated (`scoring/base.py::aggregate`). |
| `cascade_blend_weight` | `0.5` | Consumed by `reliability/cascade.py`, not by this package directly — listed here because the config file is shared. |

---

## Dependencies

Not pinned in a single `requirements.txt` today — install directly:

```
torch
transformers
datasets
spacy               # + the en_core_web_sm model (see below)
sentence-transformers
scikit-learn
scipy
PyYAML
matplotlib
pytest              # for the test suite
```

Plus the spaCy English pipeline:
```
python -m spacy download en_core_web_sm
```

Models pulled automatically from the HuggingFace Hub on first use (no manual download step): `Babelscape/rebel-large` (claim extraction), `sentence-transformers/all-MiniLM-L6-v2` (semantic matching), `potsawee/deberta-v3-large-mnli` (optional Tier-3 NLI arbitration).

---

## How to Run

### 1. Set up the environment

```bash
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

pip install torch transformers datasets spacy sentence-transformers scikit-learn scipy PyYAML matplotlib pytest
python -m spacy download en_core_web_sm
```

### 2. Build the caches (run once — GPU recommended for step 2a)

```bash
# 2a. REBEL claim extraction (GPU, slow — smoke test first)
python experiments/build_belief_cache.py --limit 5
python experiments/build_belief_cache.py --resume

# 2b. Canonicalization + semantic-embedding cache (CPU-only, fast)
python experiments/canonicalize_cache.py
```

This produces `experiments/results/claim_cache.pkl` (raw), then `experiments/results/belief_cache.pkl` and `experiments/results/belief_cache.pkl.embeddings.pkl` (canonical + embeddings). All three are gitignored (see `.gitignore`) — they must be rebuilt locally; they are not shipped in the repository.

### 3. Run the module independently (no fusion with SelfCheckGPT)

```bash
python experiments/evaluate_belief.py
```

Ablates all three scorers by default. To run a specific scorer, or to enable Tier-3 NLI arbitration for this run only:

```bash
python experiments/evaluate_belief.py --methods graph
python experiments/evaluate_belief.py --methods baseline bayesian graph --use-nli-arbitration
```

### 4. Diagnostics

```bash
python experiments/measure_match_tiers.py                       # tier/transition rate breakdown
python experiments/relation_frequency_report.py --top 40        # RELATION_MAPPING coverage
python experiments/compare_versions.py --v1 <old_cache.pkl> --v2 <new_cache.pkl>
```

### 5. Reproduce the fused (Belief Stability + SelfCheckGPT) results

Requires `experiments/results/selfcheck_cache.pkl` too (built via `python experiments/build_selfcheck_cache.py --resume`, part of the SelfCheckGPT/reliability side of the project):

```bash
python experiments/evaluate_reliability.py
```

### 6. Run the tests

```bash
python -m pytest tests/ -q
```

---

## Expected Outputs

Running the full pipeline end to end produces:

| Stage | Output file(s) |
|---|---|
| `build_belief_cache.py` | `experiments/results/claim_cache.pkl` |
| `canonicalize_cache.py` | `experiments/results/belief_cache.pkl`, `experiments/results/belief_cache.pkl.embeddings.pkl` |
| `evaluate_belief.py` | `experiments/results/belief_stability_auc.csv`, `experiments/results/pr_curve_{baseline,bayesian,graph}_nonfact.png` |
| `compare_versions.py` | `experiments/results/version_comparison.csv` |
| `evaluate_reliability.py` (fused, see the reliability engine) | `experiments/results/reliability_summary.csv`, `reliability_cv_results.csv`, `significance_tests.csv`, `subgroup_recall.png`, `pr_curve_reliability_aggregate.png`, `pr_curve_headline_comparison.png`, `alpha_selection.csv`, `alpha_distribution.png` |

## Generated CSVs/Figures

- **`belief_stability_auc.csv`** — columns `method, nonfact_auc_pr, nonfact_hard_auc_pr, factual_auc_pr, random_baseline_nonfact`, one row per scorer.
- **`pr_curve_{method}_nonfact.png`** — precision-recall curve for detecting non-factual sentences, per scorer.
- **`version_comparison.csv`** — columns `metric, v1, v2, delta` (extraction density, transition rates, tier rates, graph connectivity, and AUC-PR if available).
- **`reliability_summary.csv`** / **`reliability_cv_results.csv`** — aggregate AUC-PR and consistent-hallucination-subgroup recall, per arm (`selfcheckgpt`, `bs_baseline`, `bs_bayesian`, `bs_graph`, `cascade`), mean/std across 5 outer folds and per-fold respectively.
- **`significance_tests.csv`** — paired t-tests comparing the cascade arm against `selfcheckgpt` and `bs_graph`.

---

## How it integrates with the overall SelfCheckGPT framework

Belief Stability never runs alone in production. It is one signal inside a broader **Reliability Aggregation Engine** (`reliability/`) that fuses it with SelfCheckGPT-NLI's own resampling-consistency score:

```
reliability/features.py::build_dataset()
   reads belief_cache.pkl + selfcheck_cache.pkl
   → per-sentence SentenceRow (selfcheck_score, baseline_score, bayesian_score, support/absent/contradict)
   → per-document DocumentContext (whole-document BeliefProfiles, for GraphScorer at any alpha)

reliability/cascade.py::BeliefStabilityStage
   consumes DocumentContext, computes the graph score via GraphScorer,
   blends it with the incoming SelfCheckGPT-lineage score:
       blended = blend_weight * incoming_score + (1 - blend_weight) * risk(graph_score)
       risk(x) = (1 - x) / 2     # maps graph_score's [-1, 1] (supported..contradicted)
                                 # onto [0, 1] "risk of non-factual"
```

In the production `ReliabilityCascade` (`SelfCheckGPT -> Belief Stability -> Evidence-Grounded Verification`), `BeliefStabilityStage` sits second: rows SelfCheckGPT is already confident about never reach it; rows it escalates get a Belief-Stability-informed blended score, which is itself passed forward (via a fallback-forwarding mechanism) to the third stage if that stage abstains. See **`README_EGV.md`** for the third stage and the full cascade's semantics.

Belief Stability is also compared, as an alternative fusion strategy, against a flat logistic-regression fusion (`reliability/aggregator.py::ReliabilityAggregator`) in the `bs_baseline` / `bs_bayesian` / `bs_graph` arms of `evaluate_reliability.py` — this flat-fusion approach was found to systematically under-recall the hardest hallucination subgroup and is retained only as a documented baseline, not the recommended path for new signals.

---

## Current Results / Key Findings

From `experiments/evaluate_reliability.py` (5-fold, document-level, nested CV; see that script for the exact protocol):

| Arm | Aggregate AUC-PR | Consistent-Hallucination Subgroup Recall@0.5 |
|---|---|---|
| `selfcheckgpt` (baseline, no Belief Stability) | 92.62 ± 0.92 | 67.99 ± 3.61 |
| `bs_baseline` (LR fusion, `BaselineScorer`) | 92.40 ± 1.18 | 32.72 ± 8.33 |
| `bs_bayesian` (LR fusion, `BayesianScorer`) | 92.34 ± 1.12 | 33.37 ± 6.01 |
| `bs_graph` (LR fusion, `GraphScorer`) | 92.47 ± 1.11 | 35.34 ± 5.74 |
| **`cascade`** (SelfCheckGPT → Belief Stability, escalation cascade) | **92.68 ± 0.95** | **87.66 ± 4.46** |

Key findings:
- **Fusion architecture matters far more than which scorer is used.** All three LR-fusion arms recover only ~33–35% subgroup recall despite using the same underlying `GraphScorer`/`BayesianScorer`/`BaselineScorer` signals that the cascade arm uses — a flat logistic-regression fusion implies a decision boundary on `selfcheck_score` far above the 0.5 evaluation convention, which mechanically under-recalls the hard subgroup regardless of how good Belief Stability's own signal is.
- **The escalation cascade recovers the vast majority of the gap**: 87.66% subgroup recall vs. SelfCheckGPT alone's 67.99%, a statistically significant improvement (paired t-test across folds, `p = 0.0034`), and dramatically ahead of the best LR-fusion arm (`p = 0.0002`).
- **Belief graph propagation (`GraphScorer`) is the intended production scorer** (`scoring_method: bayesian` is actually the shipped default in `configs/default.yaml`, but `graph` is the architecturally novel arm the cascade uses via `attach_graph_scores`/`graph_scores_for_document`).

---

## Current Limitations

- **Tier 2/3 gating requires both a config flag *and* a caller-supplied instance.** `use_semantic_matching=True` and `use_nli_arbitration=False` are config defaults, but a bare `BeliefStabilityPipeline()` or `BeliefMatcher()` will not construct a `SemanticMatcher`/`NLIArbitrator` on its own — the caller must explicitly build and pass one in. `experiments/compare_versions.py` documents (in-code) a real instance of this confound having silently zeroed out Tier-2 matching in an earlier version of that script.
- **The Tier-1/fallback tier tag is overloaded**: `TransitionClassifier`'s fallback `CONTRADICT` case (candidates existed for the same subject/relation but none matched as SUPPORT) is tagged `tier=MatchTier.EXACT`, the same tag genuine Tier-1 exact matches use. Tier-distribution statistics (`measure_match_tiers.py`, `compare_versions.py`) should be read with this in mind.
- **No neural coreference resolution.** Pronoun resolution (`pronoun_resolver.py`) is a deliberately conservative, auditable heuristic (single primary-subject anchor, competing-`PERSON`-entity check) rather than a learned coreference model — it can under-resolve in documents with more complex reference chains.
- **`GraphScorer`'s adjacency can double-count a neighbor** when two beliefs share *both* their subject and object strings (once via each shared entity), giving that neighbor pair double weight in the propagated-score average.
- **REBEL extraction quality is a hard ceiling.** Any relation REBEL fails to extract, or extracts with the wrong argument order, is invisible to every downstream stage — Belief Stability cannot recover a fact it never received as a claim.
- **Entity canonicalization is heuristic, not learned**: `EntityMapper`'s global aliases and `DocumentEntityNormalizer`'s title/surname/nickname merging are curated rule tables (`constants.py`), not a trained entity-linking model; they generalize only as far as their tables cover.
- **Single-dataset validation.** All current results are on `potsawee/wiki_bio_gpt3_hallucination` (WikiBio biographies); REBEL's relation vocabulary and `constants.py`'s canonical relation set are both implicitly tuned to biography-style facts.

---

## Future Improvements

- Replace or supplement REBEL with a more modern / higher-recall open information extraction or relation-extraction model, and formally measure extraction recall against a gold claim set.
- Learn the canonicalization/entity-alias step (e.g. an entity-linking model) instead of curated lookup tables, to generalize beyond the current relation/alias vocabulary.
- Explore a fully-learned tier cascade (e.g. a single calibrated matching score with a learned decision boundary) rather than hand-tuned similarity/ambiguity-band thresholds.
- Extend the belief graph beyond single-document, single-hop propagation (e.g. multi-hop consistency reasoning across a chain of related beliefs).
- Validate on additional long-form generation / hallucination-detection datasets beyond WikiBio to test how much of the current design is dataset-specific.

---

## References

- Manakul, P., Liusie, A., & Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
- Huguet Cabot, P.-L., & Navigli, R. (2021). *REBEL: Relation Extraction By End-to-end Language generation.* Findings of EMNLP 2021. Model: [`Babelscape/rebel-large`](https://huggingface.co/Babelscape/rebel-large)
- He, P., Gao, J., & Chen, W. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* Model checkpoint used for optional Tier-3 arbitration: [`potsawee/deberta-v3-large-mnli`](https://huggingface.co/potsawee/deberta-v3-large-mnli).
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. Model used for semantic matching: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Dataset: [`potsawee/wiki_bio_gpt3_hallucination`](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination).

---

*Belief Stability is one stage of the project's broader Hybrid Hallucination Detection Framework. See `README_EGV.md` for the Evidence-Grounded Verification stage, and `reliability/` for the fusion engine that combines both with SelfCheckGPT.*
