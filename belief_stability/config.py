"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Configuration

File        : config.py

Description
-----------
Single dataclass holding every tunable knob in the Belief
Stability pipeline (extractor, matcher, scorer). Loaded
from a plain YAML file - not Hydra. This is a single
researcher's project, not a multi-node sweep; Hydra can be
layered in later if grid sweeps become necessary.
---------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class BeliefStabilityConfig:

    # ---- Extractor ----
    extractor_batch_size: int = 16
    num_beams: int = 1
    precision: str = "fp16"
    max_input_length: int = 512
    max_output_length: int = 512

    # ---- Offline cache builder ----
    flush_every: int = 10

    # ---- Extraction pre-processing ----
    use_pronoun_resolution: bool = True

    # ---- Canonicalization (ablation toggles - see
    # experiments/canonicalize_cache.py) ----
    use_document_entity_normalization: bool = True
    use_inverse_matching: bool = True

    # ---- Matcher ----
    use_semantic_matching: bool = True
    match_similarity_threshold: float = 0.82
    use_nli_arbitration: bool = False
    nli_ambiguous_low: float = 0.55
    nli_ambiguous_high: float = 0.82

    # ---- Scoring ----
    scoring_method: str = "bayesian"  # "baseline" | "bayesian" | "graph"
    absent_discount: float = 0.5
    graph_alpha: float = 0.3

    # ---- Sentence aggregation (belief scores -> sentence score) ----
    sentence_aggregation: str = "min"  # "min" | "mean"

    # ---- Reliability Cascade (reliability/cascade.py) ----
    cascade_blend_weight: float = 0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BeliefStabilityConfig":

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        known_fields = cls.__dataclass_fields__.keys()

        filtered = {
            k: v for k, v in raw.items() if k in known_fields
        }

        return cls(**filtered)

    def to_yaml(self, path: str | Path) -> None:

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
