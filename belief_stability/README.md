# Belief Stability Module Design Specification (Version 1)

## Project

Hybrid Hallucination Detection and Reliability Scoring Framework for Large Language Models

---

# 1. Purpose

The objective of this module is to measure **belief stability** across multiple stochastic generations of an LLM.

Unlike SelfCheckGPT, which primarily measures textual consistency, this module operates at the **claim level** and models how individual factual beliefs behave across multiple generated passages.

This module does **not** verify factual correctness.

Instead, it evaluates whether the model consistently maintains or changes its expressed beliefs.

---

# 2. Position in Overall Framework

```
Dataset
    │
    ▼
GPT-3 Generated Passage
    │
    ▼
20 Sampled Passages
    │
    ▼
SelfCheckGPT
    │
    ▼
Belief Stability Module
    │
    ▼
Counterfactual Verification (Future)
    │
    ▼
Adversarial Verification (Future)
    │
    ▼
Reliability Aggregation
```

This document only describes the **Belief Stability Module**.

---

# 3. Research Motivation

SelfCheckGPT assumes that consistent generations imply factual reliability.

However,

Consistency ≠ Truth.

Large Language Models may repeatedly generate the same incorrect information.

Therefore, instead of measuring only textual consistency, this module models the persistence of individual factual beliefs.

---

# 4. Module Pipeline

```
Original Passage
        │
        ▼
Claim Extraction
        │
        ▼
Belief Canonicalization
        │
        ▼
Canonical Beliefs
        │
        ▼
Belief Lookup
        │
        ▼
Transition Classification
        │
        ▼
Transition Profile
        │
        ▼
Belief Stability Score
```

---

# 5. Scope

This module is responsible only for:

* Claim-level analysis
* Belief representation
* Belief matching
* Transition classification
* Belief stability computation

This module does **not**:

* verify truth
* retrieve external knowledge
* perform counterfactual reasoning
* use adversarial prompting

---

# 6. Claim Extraction

Claim extraction is considered an external component.

The framework is extractor-agnostic.

Version 1 uses REBEL as the default extractor.

The extractor may be replaced in future without modifying the remainder of the framework.

---

# 7. Belief Canonicalization

Purpose:

Convert extractor-specific outputs into a unified internal representation.

## 7.1 Preprocessing

* whitespace cleanup
* punctuation cleanup
* formatting cleanup

No semantic reasoning is performed.

---

## 7.2 Normalization

### Relation Normalization

Example

```
founder_of
↓

FOUNDED
```

```
birthplace
↓

BORN_IN
```

---

### Entity Normalization

Example

```
USA
↓

United States
```

```
U.S.
↓

United States
```

---

### Attribute Normalization

Example

```
Founded in 1976
```

↓

```
attributes:

year = 1976
```

---

## 7.3 Belief Construction

The canonicalizer creates a standard Belief object.

---

# 8. Canonical Belief Representation

Each belief is represented as

```
Belief

subject

relation

object

attributes

source_text

confidence
```

where

* subject : entity
* relation : canonical relation
* object : entity/value
* attributes : optional qualifiers
* source_text : extracted claim
* confidence : extractor confidence (optional)

---

# 9. Belief Lookup

For every sampled passage, beliefs are indexed using

```
(subject, relation)
```

Example

```
("Elon Musk", "BORN_IN")

↓

[
Belief(...)
]
```

Lookup returns all candidate beliefs corresponding to the original belief.

No embedding similarity or LLM reasoning is performed.

---

# 10. Transition Classification

Each original belief is assigned one of three transition states.

## Support

The sampled passage expresses the same belief.

```
Original

Elon
BORN_IN
South Africa

↓

Sample

Elon
BORN_IN
South Africa
```

---

## Contradict

The sampled passage expresses an incompatible belief.

```
Original

Elon
BORN_IN
South Africa

↓

Sample

Elon
BORN_IN
Canada
```

---

## Absent

The sampled passage contains no matching belief for the corresponding subject–relation pair.

Example

Original

```
Elon
BORN_IN
South Africa
```

Sample

```
Elon
CEO_OF
Tesla
```

No birthplace information is present.

Transition:

Absent

---

# 11. Transition Profile

For each original belief, transitions are recorded across all sampled passages.

Example

```
Belief

Steve Jobs
FOUNDED
Apple

Support      : 18

Absent       : 1

Contradict   : 1
```

---

# 12. Belief Stability Score

The transition profile is summarized into a scalar Belief Stability Score.

The exact mathematical formulation is defined separately during implementation.

The score reflects how consistently the model maintains a belief across stochastic generations.

---

# 13. Design Principles

The module follows the following principles:

* Claim-centric analysis
* Deterministic processing
* Explainable decisions
* Modular architecture
* Extractor independence
* Reproducibility
* No external knowledge retrieval
* No additional LLM reasoning
* Easy integration with SelfCheckGPT

---

# 14. Future Extensions

The following components are intentionally outside the scope of Version 1:

* Counterfactual Verification
* Adversarial Verification
* Reliability Aggregation
* External Knowledge Verification
* Belief Refinement Detection

These are implemented as independent downstream modules.
