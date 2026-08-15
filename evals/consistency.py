"""Consistency Testing Evaluator.

Asks the same factual question in 5 different phrasings and checks
whether answers are semantically consistent using token overlap.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from config import TEST_CASES_DIR
from llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    test_id: str
    topic: str
    variants: list[str]
    answers: list[str]
    pair_scores: list[dict]
    consistency_score: float
    has_contradiction: bool


@dataclass
class ConsistencyReport:
    total_topics: int
    avg_consistency: float
    contradictions_found: int
    consistency_score: float
    results: list[ConsistencyResult]
    failures: list[ConsistencyResult] = field(default_factory=list)


def load_test_cases() -> list[dict]:
    path = TEST_CASES_DIR / "consistency.json"
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


def token_overlap_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard-like token overlap between two texts."""
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def extract_key_facts(text: str) -> set[str]:
    """Extract numbers, proper nouns, and key terms from an answer."""
    facts = set()
    # Numbers (including decimals and commas)
    numbers = re.findall(r"\b[\d,]+\.?\d*\b", text)
    facts.update(numbers)
    # Capitalized words (likely proper nouns / key terms)
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    facts.update(w.lower() for w in caps)
    return facts


def check_contradiction(answers: list[str]) -> tuple[float, list[dict], bool]:
    """Compare all pairs of answers for consistency.

    Returns (avg_similarity, pair_details, has_contradiction).
    """
    pairs = list(combinations(range(len(answers)), 2))
    pair_scores = []
    total_sim = 0.0

    for i, j in pairs:
        sim = token_overlap_similarity(answers[i], answers[j])
        facts_i = extract_key_facts(answers[i])
        facts_j = extract_key_facts(answers[j])
        fact_overlap = len(facts_i & facts_j) / max(len(facts_i | facts_j), 1)

        # Weighted: 60% token overlap + 40% fact overlap
        combined = 0.6 * sim + 0.4 * fact_overlap

        pair_scores.append({
            "pair": (i, j),
            "token_similarity": round(sim, 3),
            "fact_overlap": round(fact_overlap, 3),
            "combined_score": round(combined, 3),
        })
        total_sim += combined

    avg_sim = total_sim / len(pairs) if pairs else 1.0
    has_contradiction = any(p["combined_score"] < 0.15 for p in pair_scores)

    return avg_sim, pair_scores, has_contradiction


def run(client: LLMClient) -> ConsistencyReport:
    """Run the consistency testing suite."""
    test_cases = load_test_cases()
    results: list[ConsistencyResult] = []

    for tc in test_cases:
        logger.info("Consistency test %s: %s", tc["id"], tc["topic"])
        answers = []

        for variant in tc["variants"]:
            try:
                response: LLMResponse = client.generate(
                    prompt=f"Answer this question concisely:\n\n{variant}",
                    temperature=0.0,
                )
                answers.append(response.text.strip())
            except Exception as e:
                logger.error("Error on variant %r: %s", variant, e)
                answers.append(f"ERROR: {e}")

        consistency, pair_scores, has_contradiction = check_contradiction(answers)

        result = ConsistencyResult(
            test_id=tc["id"],
            topic=tc["topic"],
            variants=tc["variants"],
            answers=answers,
            pair_scores=pair_scores,
            consistency_score=round(consistency, 3),
            has_contradiction=has_contradiction,
        )
        results.append(result)

    avg_consistency = (
        sum(r.consistency_score for r in results) / len(results)
        if results
        else 0
    )
    contradictions = sum(1 for r in results if r.has_contradiction)
    failures = [r for r in results if r.has_contradiction]

    return ConsistencyReport(
        total_topics=len(results),
        avg_consistency=round(avg_consistency, 3),
        contradictions_found=contradictions,
        consistency_score=round(avg_consistency, 3),
        results=results,
        failures=failures,
    )
