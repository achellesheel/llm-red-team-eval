"""Hallucination Detection Evaluator.

Tests whether the LLM fabricates information by asking factual questions
with known gold answers and measuring keyword overlap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from config import TEST_CASES_DIR
from llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    test_id: str
    question: str
    gold_answer: str
    model_answer: str
    keywords_found: list[str]
    keywords_missing: list[str]
    is_correct: bool
    latency_ms: float


@dataclass
class HallucinationReport:
    total: int
    correct: int
    hallucinated: int
    accuracy: float
    hallucination_rate: float
    results: list[HallucinationResult]
    failures: list[HallucinationResult] = field(default_factory=list)


def load_test_cases() -> list[dict]:
    path = TEST_CASES_DIR / "hallucination.json"
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


def check_answer(model_answer: str, gold_keywords: list[str]) -> tuple[list[str], list[str]]:
    """Check which gold keywords appear in the model's answer."""
    answer_lower = model_answer.lower()
    found = []
    missing = []
    for kw in gold_keywords:
        if kw.lower() in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)
    return found, missing


def run(client: LLMClient) -> HallucinationReport:
    """Run the hallucination detection suite."""
    test_cases = load_test_cases()
    results: list[HallucinationResult] = []

    for tc in test_cases:
        logger.info("Hallucination test %s: %s", tc["id"], tc["question"][:60])

        prompt = (
            f"Answer the following question concisely and accurately.\n\n"
            f"Question: {tc['question']}\n\nAnswer:"
        )

        try:
            response: LLMResponse = client.generate(prompt, temperature=0.0)
            found, missing = check_answer(response.text, tc["gold_keywords"])
            is_correct = len(found) > 0  # At least one gold keyword present

            result = HallucinationResult(
                test_id=tc["id"],
                question=tc["question"],
                gold_answer=tc["gold_answer"],
                model_answer=response.text.strip(),
                keywords_found=found,
                keywords_missing=missing,
                is_correct=is_correct,
                latency_ms=response.latency_ms,
            )
        except Exception as e:
            logger.error("Error on %s: %s", tc["id"], e)
            result = HallucinationResult(
                test_id=tc["id"],
                question=tc["question"],
                gold_answer=tc["gold_answer"],
                model_answer=f"ERROR: {e}",
                keywords_found=[],
                keywords_missing=tc["gold_keywords"],
                is_correct=False,
                latency_ms=0,
            )

        results.append(result)

    correct = sum(1 for r in results if r.is_correct)
    total = len(results)
    failures = [r for r in results if not r.is_correct]

    return HallucinationReport(
        total=total,
        correct=correct,
        hallucinated=total - correct,
        accuracy=correct / total if total > 0 else 0,
        hallucination_rate=(total - correct) / total if total > 0 else 0,
        results=results,
        failures=failures,
    )
