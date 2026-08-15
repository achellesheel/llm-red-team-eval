"""Eval Runner — orchestrates all evaluation suites and produces a Model Report Card.

Usage:
    python runner.py                        # Run all suites, save report
    python runner.py --suites hallucination injection
    python runner.py --provider ollama --model llama3.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from config import CATEGORY_WEIGHTS, PASS_THRESHOLDS, RESULTS_DIR
from llm_client import LLMClient

from evals import hallucination, injection, consistency, refusal, format_compliance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUITE_REGISTRY = {
    "hallucination": hallucination,
    "injection": injection,
    "consistency": consistency,
    "refusal": refusal,
    "format_compliance": format_compliance,
}


def score_for_suite(suite_name: str, report) -> float:
    """Extract the 0–1 score from a suite's report object."""
    if suite_name == "hallucination":
        return report.accuracy
    elif suite_name == "injection":
        return report.resistance_rate
    elif suite_name == "consistency":
        return report.consistency_score
    elif suite_name == "refusal":
        return report.appropriate_refusal_rate
    elif suite_name == "format_compliance":
        return report.compliance_rate
    return 0.0


def run_all(
    client: LLMClient,
    suites: list[str] | None = None,
) -> dict:
    """Run specified suites (or all) and return the report card."""
    suites_to_run = suites or list(SUITE_REGISTRY.keys())
    results = {}
    scores = {}

    for name in suites_to_run:
        module = SUITE_REGISTRY.get(name)
        if module is None:
            logger.warning("Unknown suite: %s — skipping", name)
            continue

        logger.info("=" * 60)
        logger.info("Running suite: %s", name)
        logger.info("=" * 60)

        report = module.run(client)
        score = score_for_suite(name, report)
        passed = score >= PASS_THRESHOLDS.get(name, 0.7)

        scores[name] = {
            "score": round(score, 3),
            "passed": passed,
            "threshold": PASS_THRESHOLDS.get(name, 0.7),
        }

        # Serialize report (pick key fields, skip full results for summary)
        report_dict = asdict(report)
        # Keep only failures for the summary to avoid huge JSON
        summary = {k: v for k, v in report_dict.items() if k != "results"}
        summary["failures"] = [
            {k: v for k, v in f.items() if k != "pair_scores"}
            for f in summary.get("failures", [])
        ][:5]  # Top 5 failures
        results[name] = summary

    # Overall score (weighted average)
    weighted_sum = 0.0
    weight_total = 0.0
    for name, info in scores.items():
        w = CATEGORY_WEIGHTS.get(name, 0.2)
        weighted_sum += info["score"] * w
        weight_total += w

    overall_score = round(weighted_sum / weight_total, 3) if weight_total > 0 else 0

    report_card = {
        "model": client.model,
        "provider": client.provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_score": overall_score,
        "overall_passed": all(s["passed"] for s in scores.values()),
        "category_scores": scores,
        "category_details": results,
        "weights": CATEGORY_WEIGHTS,
    }

    return report_card


def save_report(report_card: dict) -> Path:
    """Save report card to JSON file."""
    model_name = report_card["model"].replace("/", "_").replace(":", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"report_{model_name}_{ts}.json"
    path = RESULTS_DIR / filename

    with open(path, "w") as f:
        json.dump(report_card, f, indent=2, default=str)

    logger.info("Report saved to %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="LLM Red-Team Eval Runner")
    parser.add_argument(
        "--provider",
        choices=["groq", "ollama", "openai_compatible"],
        default=None,
        help="LLM provider (overrides LLM_PROVIDER env var)",
    )
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="Base URL")
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=list(SUITE_REGISTRY.keys()),
        default=None,
        help="Specific suites to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/<model>_<timestamp>.json)",
    )

    args = parser.parse_args()

    client = LLMClient(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    logger.info("Using %s", client)

    report_card = run_all(client, args.suites)

    if args.output:
        path = Path(args.output)
        with open(path, "w") as f:
            json.dump(report_card, f, indent=2, default=str)
        logger.info("Report saved to %s", path)
    else:
        path = save_report(report_card)

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL REPORT CARD")
    print("=" * 60)
    print(f"Model:   {report_card['model']}")
    print(f"Overall: {report_card['overall_score']:.1%} {'✅ PASS' if report_card['overall_passed'] else '❌ FAIL'}")
    print("-" * 40)
    for cat, info in report_card["category_scores"].items():
        status = "✅" if info["passed"] else "❌"
        print(f"  {cat:20s} {info['score']:.1%}  {status}  (threshold: {info['threshold']:.0%})")
    print("=" * 60)
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
