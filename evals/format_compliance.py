"""Format Compliance Evaluator.

Tests whether the model follows specific output format instructions
(JSON, markdown tables, bullet lists, CSV, etc.).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from config import TEST_CASES_DIR
from llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class FormatResult:
    test_id: str
    prompt: str
    expected_format: str
    model_response: str
    is_compliant: bool
    issues: list[str]
    latency_ms: float


@dataclass
class FormatReport:
    total: int
    compliant: int
    non_compliant: int
    compliance_rate: float
    results: list[FormatResult]
    failures: list[FormatResult] = field(default_factory=list)


def load_test_cases() -> list[dict]:
    path = TEST_CASES_DIR / "format_compliance.json"
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


# ── Validators ──────────────────────────────────────────────────────────


def _extract_json(text: str) -> str:
    """Try to extract JSON from a response that might have extra text."""
    # Try raw first
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def validate_json(response: str, validation: dict) -> tuple[bool, list[str]]:
    issues = []
    cleaned = _extract_json(response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    if "min_items" in validation:
        if isinstance(data, list):
            if len(data) < validation["min_items"]:
                issues.append(
                    f"Expected >= {validation['min_items']} items, got {len(data)}"
                )
        elif isinstance(data, dict) and "items" in data:
            if isinstance(data["items"], list) and len(data["items"]) < validation.get("min_items", 0):
                issues.append(f"Too few items in 'items' array")

    if "required_keys" in validation:
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    for key in validation["required_keys"]:
                        if key not in item:
                            issues.append(f"Item {i} missing key '{key}'")
                    break  # Only check first item
        elif isinstance(data, dict):
            for key in validation["required_keys"]:
                if key not in data:
                    issues.append(f"Missing key '{key}'")

    return len(issues) == 0, issues


def validate_markdown_table(response: str, validation: dict) -> tuple[bool, list[str]]:
    issues = []
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]

    # Find table lines (contain |)
    table_lines = [l for l in lines if "|" in l]
    if len(table_lines) < 2:
        return False, ["No markdown table found"]

    # Check headers
    header_line = table_lines[0]
    if "required_headers" in validation:
        for h in validation["required_headers"]:
            if h.lower() not in header_line.lower():
                issues.append(f"Missing header '{h}'")

    # Check row count (exclude header and separator)
    data_rows = [l for l in table_lines if not re.match(r"^[\|\s\-:]+$", l)]
    data_rows = data_rows[1:]  # Remove header row
    if "min_rows" in validation and len(data_rows) < validation["min_rows"]:
        issues.append(
            f"Expected >= {validation['min_rows']} data rows, got {len(data_rows)}"
        )

    return len(issues) == 0, issues


def validate_bullet_list(response: str, validation: dict) -> tuple[bool, list[str]]:
    issues = []
    char = validation.get("bullet_char", "-")
    bullets = [l.strip() for l in response.strip().split("\n") if l.strip().startswith(char)]

    if "exact_count" in validation and len(bullets) != validation["exact_count"]:
        issues.append(
            f"Expected {validation['exact_count']} bullets, got {len(bullets)}"
        )

    return len(issues) == 0, issues


def validate_numbered_list(response: str, validation: dict) -> tuple[bool, list[str]]:
    issues = []
    numbered = re.findall(r"^\d+\.\s+.+", response.strip(), re.MULTILINE)

    if "exact_count" in validation and len(numbered) != validation["exact_count"]:
        issues.append(
            f"Expected {validation['exact_count']} numbered items, got {len(numbered)}"
        )

    return len(issues) == 0, issues


def validate_csv(response: str, validation: dict) -> tuple[bool, list[str]]:
    issues = []
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        reader = csv.reader(io.StringIO(cleaned))
        rows = list(reader)
    except Exception as e:
        return False, [f"Invalid CSV: {e}"]

    if len(rows) < 2:
        return False, ["CSV has fewer than 2 rows (header + data)"]

    if "required_headers" in validation:
        headers = [h.strip() for h in rows[0]]
        for h in validation["required_headers"]:
            if h not in headers:
                issues.append(f"Missing CSV header '{h}'")

    data_rows = rows[1:]
    if "min_rows" in validation and len(data_rows) < validation["min_rows"]:
        issues.append(
            f"Expected >= {validation['min_rows']} data rows, got {len(data_rows)}"
        )

    return len(issues) == 0, issues


def validate_line_count(response: str, validation: dict) -> tuple[bool, list[str]]:
    issues = []
    lines = [l for l in response.strip().split("\n") if l.strip()]
    if "exact_lines" in validation and len(lines) != validation["exact_lines"]:
        issues.append(
            f"Expected {validation['exact_lines']} lines, got {len(lines)}"
        )
    return len(issues) == 0, issues


def validate_python_code(response: str, validation: dict) -> tuple[bool, list[str]]:
    issues = []
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    for s in validation.get("required_strings", []):
        if s not in cleaned:
            issues.append(f"Missing required string '{s}'")

    # Try to compile
    try:
        compile(cleaned, "<string>", "exec")
    except SyntaxError as e:
        issues.append(f"Python syntax error: {e}")

    return len(issues) == 0, issues


def validate_exact_match(response: str, validation: dict) -> tuple[bool, list[str]]:
    expected = validation["expected"]
    actual = response.strip()
    if actual == expected:
        return True, []
    return False, [f"Expected exact '{expected}', got '{actual}'"]


VALIDATORS = {
    "json": validate_json,
    "markdown_table": validate_markdown_table,
    "bullet_list": validate_bullet_list,
    "numbered_list": validate_numbered_list,
    "csv": validate_csv,
    "line_count": validate_line_count,
    "python_code": validate_python_code,
    "exact_match": validate_exact_match,
}


# ── Runner ──────────────────────────────────────────────────────────────


def run(client: LLMClient) -> FormatReport:
    """Run the format compliance suite."""
    test_cases = load_test_cases()
    results: list[FormatResult] = []

    for tc in test_cases:
        logger.info("Format test %s: %s", tc["id"], tc["expected_format"])
        validation = tc["validation"]
        validator_fn = VALIDATORS.get(validation["type"])

        try:
            response: LLMResponse = client.generate(
                prompt=tc["prompt"],
                temperature=0.0,
            )

            if validator_fn:
                is_compliant, issues = validator_fn(response.text, validation)
            else:
                is_compliant, issues = False, [f"Unknown validator: {validation['type']}"]

            result = FormatResult(
                test_id=tc["id"],
                prompt=tc["prompt"],
                expected_format=tc["expected_format"],
                model_response=response.text.strip(),
                is_compliant=is_compliant,
                issues=issues,
                latency_ms=response.latency_ms,
            )
        except Exception as e:
            logger.error("Error on %s: %s", tc["id"], e)
            result = FormatResult(
                test_id=tc["id"],
                prompt=tc["prompt"],
                expected_format=tc["expected_format"],
                model_response=f"ERROR: {e}",
                is_compliant=False,
                issues=[str(e)],
                latency_ms=0,
            )

        results.append(result)

    compliant = sum(1 for r in results if r.is_compliant)
    total = len(results)
    failures = [r for r in results if not r.is_compliant]

    return FormatReport(
        total=total,
        compliant=compliant,
        non_compliant=total - compliant,
        compliance_rate=compliant / total if total > 0 else 0,
        results=results,
        failures=failures,
    )
