"""
---------------------------------------------------------
Belief Stability Module

Author      : Sujay Bhat
Project     : Hybrid Hallucination Detection Framework
Component   : Claim Extraction

File        : claim_extractor.py

Description
-----------
Claim extractor based on REBEL, batched for GPU throughput.

Pipeline

Raw Texts (batch)
    ↓
REBEL (padded batch, fp16 autocast on CUDA)
    ↓
(subject, relation, object) triplets per text
    ↓
ExtractedClaim objects

``extract_batch`` is the primary entry point: it pads the
whole batch through a single forward pass instead of one
text at a time, which is what makes offline cache-building
over thousands of texts tractable. ``extract`` is a thin
single-text convenience wrapper kept for interactive/demo
use.
---------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from belief_stability.models import ExtractedClaim


class ClaimExtractor(ABC):
    """
    Interface every claim extractor must implement.

    Keeping extraction behind this interface is what makes
    the framework extractor-agnostic (REBEL today, GLiREL /
    spaCy OpenIE / others later) without touching anything
    downstream of ``ExtractedClaim``.
    """

    @abstractmethod
    def extract_batch(
        self,
        texts: List[str],
    ) -> List[List[ExtractedClaim]]:
        raise NotImplementedError

    def extract(
        self,
        text: str,
    ) -> List[ExtractedClaim]:

        if text is None:
            return []

        text = text.strip()

        if len(text) == 0:
            return []

        return self.extract_batch([text])[0]


class RebelClaimExtractor(ClaimExtractor):

    MODEL_NAME = "Babelscape/rebel-large"

    def __init__(
        self,
        device: str | None = None,
        precision: str = "fp16",
        batch_size: int = 16,
        num_beams: int = 1,
        max_input_length: int = 512,
        max_output_length: int = 512,
    ) -> None:

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        self.batch_size = batch_size

        self.num_beams = num_beams

        self.max_input_length = max_input_length

        self.max_output_length = max_output_length

        # fp16 only makes sense on CUDA; CPU always runs fp32.
        self.use_fp16 = (
            precision == "fp16"
            and self.device == "cuda"
        )

        print("Loading REBEL tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        print("Loading REBEL model...")

        self.model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(self.MODEL_NAME)
            .to(self.device)
        )

        if self.use_fp16:
            self.model = self.model.half()

        self.model.eval()

    # --------------------------------------------------

    @torch.inference_mode()
    def _generate_batch(
        self,
        texts: List[str],
    ) -> List[str]:

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_input_length,
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        generated_tokens = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            num_beams=self.num_beams,
            early_stopping=True,
        )

        return self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=False,
        )

    # --------------------------------------------------

    def _parse_triplets(
        self,
        text: str,
    ) -> List[Dict]:

        triplets = []

        relation = ""

        subject = ""

        object_ = ""

        current = None

        text = (
            text.replace("<s>", "")
                .replace("</s>", "")
                .replace("<pad>", "")
        )

        for token in text.split():

            if token == "<triplet>":

                current = "t"

                if relation != "":

                    triplets.append(
                        {
                            "subject": subject.strip(),
                            "relation": relation.strip(),
                            "object": object_.strip(),
                        }
                    )

                    relation = ""

                subject = ""

            elif token == "<subj>":

                current = "s"

                if relation != "":

                    triplets.append(
                        {
                            "subject": subject.strip(),
                            "relation": relation.strip(),
                            "object": object_.strip(),
                        }
                    )

                object_ = ""

            elif token == "<obj>":

                current = "o"

                relation = ""

            else:

                if current == "t":

                    subject += " " + token

                elif current == "s":

                    object_ += " " + token

                elif current == "o":

                    relation += " " + token

        if (
            subject.strip()
            and relation.strip()
            and object_.strip()
        ):

            triplets.append(
                {
                    "subject": subject.strip(),
                    "relation": relation.strip(),
                    "object": object_.strip(),
                }
            )

        return triplets

    # --------------------------------------------------

    def _convert_to_claims(
        self,
        triplets: List[Dict],
        source_text: str,
    ) -> List[ExtractedClaim]:

        claims: List[ExtractedClaim] = []

        for triplet in triplets:

            subject = triplet["subject"].strip()
            relation = triplet["relation"].strip()
            object_ = triplet["object"].strip()

            if (
                not subject
                or not relation
                or not object_
            ):
                continue

            claims.append(

                ExtractedClaim(
                    subject=subject,
                    relation=relation,
                    object=object_,
                    attributes={},
                    confidence=1.0,
                )

            )

        return claims

    # --------------------------------------------------

    def extract_batch(
        self,
        texts: List[str],
    ) -> List[List[ExtractedClaim]]:
        """
        Extract claims for many texts, batched through REBEL.

        Empty/whitespace-only texts are skipped (mapped to an
        empty claim list) without wasting a model call.
        """

        results: List[List[ExtractedClaim]] = [[] for _ in texts]

        job_indices = [
            i for i, t in enumerate(texts)
            if t is not None and t.strip()
        ]

        for start in range(0, len(job_indices), self.batch_size):

            chunk_indices = job_indices[start:start + self.batch_size]

            chunk_texts = [texts[i].strip() for i in chunk_indices]

            decoded = self._generate_batch(chunk_texts)

            for idx, decoded_text, source_text in zip(
                chunk_indices, decoded, chunk_texts
            ):

                triplets = self._parse_triplets(decoded_text)

                results[idx] = self._convert_to_claims(
                    triplets,
                    source_text=source_text,
                )

        return results
