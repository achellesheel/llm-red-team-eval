"""Streamlit Dashboard — Model Report Card viewer with comparison support."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from config import RESULTS_DIR

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Red-Team Eval",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ LLM Red-Team Eval — Model Report Card")
st.caption(
    "Automated safety & quality evaluation: hallucination, prompt injection, "
    "consistency, refusal appropriateness, and format compliance."
)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_reports() -> list[dict]:
    """Load all saved report JSON files."""
    reports = []
    if not RESULTS_DIR.exists():
        return reports
    for f in sorted(RESULTS_DIR.glob("report_*.json"), reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data["_file"] = f.name
                reports.append(data)
        except Exception:
            continue
    return reports


def score_color(score: float, threshold: float) -> str:
    if score >= threshold:
        return "green"
    elif score >= threshold * 0.8:
        return "orange"
    return "red"


def score_emoji(passed: bool) -> str:
    return "✅" if passed else "❌"


CATEGORY_LABELS = {
    "hallucination": "🔍 Hallucination Detection",
    "injection": "💉 Prompt Injection Resistance",
    "consistency": "🔄 Consistency",
    "refusal": "🚫 Refusal Appropriateness",
    "format_compliance": "📋 Format Compliance",
}


# ── Load data ────────────────────────────────────────────────────────────

reports = load_reports()

if not reports:
    st.warning(
        "No reports found. Run the evaluation first:\n\n"
        "```bash\npython runner.py\n```\n\n"
        "Reports are saved to `results/`."
    )
    st.stop()


# ── Sidebar: report selection ────────────────────────────────────────────

with st.sidebar:
    st.header("Reports")

    report_labels = {
        r["_file"]: f"{r['model']} — {r['timestamp'][:10]}"
        for r in reports
    }

    selected_file = st.selectbox(
        "Select report",
        options=list(report_labels.keys()),
        format_func=lambda k: report_labels[k],
    )
    report = next(r for r in reports if r["_file"] == selected_file)

    st.divider()

    # Comparison mode
    compare_enabled = st.checkbox("Compare with another report")
    compare_report = None
    if compare_enabled and len(reports) > 1:
        other_files = [f for f in report_labels if f != selected_file]
        compare_file = st.selectbox(
            "Compare against",
            options=other_files,
            format_func=lambda k: report_labels[k],
        )
        compare_report = next(r for r in reports if r["_file"] == compare_file)

    st.divider()
    st.markdown(
        "**How to run:**\n\n"
        "```bash\n"
        "python runner.py\n"
        "# or specific suites:\n"
        "python runner.py \\\n"
        "  --suites hallucination \\\n"
        "  injection\n"
        "```"
    )


# ── Overall Score ────────────────────────────────────────────────────────

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    overall = report["overall_score"]
    st.metric(
        label="Overall Safety Score",
        value=f"{overall:.1%}",
        delta=None,
    )

with col2:
    st.metric("Model", report["model"])

with col3:
    status = "PASS ✅" if report["overall_passed"] else "FAIL ❌"
    st.metric("Status", status)

st.divider()

# ── Per-Category Scores ─────────────────────────────────────────────────

st.subheader("Category Scores")

cols = st.columns(len(report["category_scores"]))
for idx, (cat, info) in enumerate(report["category_scores"].items()):
    with cols[idx]:
        label = CATEGORY_LABELS.get(cat, cat)
        emoji = score_emoji(info["passed"])

        delta = None
        if compare_report and cat in compare_report.get("category_scores", {}):
            diff = info["score"] - compare_report["category_scores"][cat]["score"]
            delta = f"{diff:+.1%}"

        st.metric(
            label=f"{emoji} {label}",
            value=f"{info['score']:.1%}",
            delta=delta,
            help=f"Threshold: {info['threshold']:.0%}",
        )

st.divider()

# ── Detailed Results per Category ────────────────────────────────────────

st.subheader("Detailed Results")

for cat, details in report["category_details"].items():
    label = CATEGORY_LABELS.get(cat, cat)
    score_info = report["category_scores"].get(cat, {})
    emoji = score_emoji(score_info.get("passed", False))

    with st.expander(f"{emoji} {label} — {score_info.get('score', 0):.1%}", expanded=False):

        # Category-specific summary metrics
        if cat == "hallucination":
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Questions", details.get("total", 0))
            c2.metric("Correct", details.get("correct", 0))
            c3.metric("Hallucinated", details.get("hallucinated", 0))

        elif cat == "injection":
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Attacks", details.get("total", 0))
            c2.metric("Resisted", details.get("resisted", 0))
            c3.metric("Compromised", details.get("compromised", 0))

            if details.get("by_attack_type"):
                st.markdown("**By Attack Type:**")
                for atype, stats in details["by_attack_type"].items():
                    rate = stats.get("resistance_rate", 0)
                    st.write(f"  - `{atype}`: {rate:.0%} resistance ({stats['resisted']}/{stats['total']})")

        elif cat == "consistency":
            c1, c2 = st.columns(2)
            c1.metric("Topics Tested", details.get("total_topics", 0))
            c2.metric("Contradictions", details.get("contradictions_found", 0))

        elif cat == "refusal":
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Queries", details.get("total", 0))
            c2.metric("False Refusals", details.get("false_refusals", 0))
            c3.metric("Missed Refusals", details.get("missed_refusals", 0))

        elif cat == "format_compliance":
            c1, c2 = st.columns(2)
            c1.metric("Compliant", details.get("compliant", 0))
            c2.metric("Non-Compliant", details.get("non_compliant", 0))

        # Show example failures
        failures = details.get("failures", [])
        if failures:
            st.markdown("**Example Failures:**")
            for i, fail in enumerate(failures[:3]):
                with st.container():
                    if cat == "hallucination":
                        st.markdown(f"**Q:** {fail.get('question', 'N/A')}")
                        st.markdown(f"**Expected:** {fail.get('gold_answer', 'N/A')}")
                        st.markdown(f"**Got:** {fail.get('model_answer', 'N/A')[:200]}")
                    elif cat == "injection":
                        st.markdown(f"**Attack ({fail.get('attack_type', '')}):** {fail.get('attack', 'N/A')[:150]}")
                        st.markdown(f"**Response:** {fail.get('model_response', 'N/A')[:200]}")
                        st.markdown(f"**Indicators found:** {fail.get('indicators_found', [])}")
                    elif cat == "consistency":
                        st.markdown(f"**Topic:** {fail.get('topic', 'N/A')}")
                        for j, ans in enumerate(fail.get("answers", [])[:3]):
                            st.markdown(f"  - Variant {j+1}: {ans[:150]}")
                    elif cat == "refusal":
                        st.markdown(f"**Query:** {fail.get('query', 'N/A')}")
                        st.markdown(f"**Should refuse:** {fail.get('should_refuse')} | **Did refuse:** {fail.get('did_refuse')}")
                        st.markdown(f"**Response:** {fail.get('model_response', 'N/A')[:200]}")
                    elif cat == "format_compliance":
                        st.markdown(f"**Format:** {fail.get('expected_format', 'N/A')}")
                        st.markdown(f"**Issues:** {fail.get('issues', [])}")
                        st.markdown(f"**Response:** {fail.get('model_response', 'N/A')[:200]}")
                    if i < len(failures) - 1:
                        st.markdown("---")
        else:
            st.success("No failures! All tests passed.")


# ── Comparison View ──────────────────────────────────────────────────────

if compare_report:
    st.divider()
    st.subheader("📊 Model Comparison")

    comparison_data = []
    for cat in report["category_scores"]:
        row = {
            "Category": CATEGORY_LABELS.get(cat, cat),
            f"{report['model']}": report["category_scores"][cat]["score"],
        }
        if cat in compare_report.get("category_scores", {}):
            row[f"{compare_report['model']}"] = compare_report["category_scores"][cat]["score"]
        comparison_data.append(row)

    st.table(comparison_data)

    # Overall comparison
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            f"{report['model']} Overall",
            f"{report['overall_score']:.1%}",
        )
    with c2:
        st.metric(
            f"{compare_report['model']} Overall",
            f"{compare_report['overall_score']:.1%}",
        )


# ── Raw JSON ─────────────────────────────────────────────────────────────

with st.expander("📄 Raw Report JSON"):
    display = {k: v for k, v in report.items() if k != "_file"}
    st.json(display)
