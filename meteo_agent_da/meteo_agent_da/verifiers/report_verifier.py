from __future__ import annotations

from typing import Any, Dict


def verify_report_format(report: Dict[str, Any]) -> Dict[str, Any]:
    summary = str(report.get("summary", "") or "")
    tool_results = list(report.get("tool_results", []) or [])
    artifacts = list(report.get("artifacts", []) or [])
    next_steps = list(report.get("next_steps", []) or [])
    text = summary.lower()

    checks = {
        "has_summary": bool(summary.strip()),
        "has_tool_evidence": bool(tool_results),
        "has_status": str(report.get("status", "")) in {"ok", "error", "skipped"},
        "has_artifact_reference": bool(artifacts) or "artifact" in text or "产物" in summary,
        "has_limitation_or_next_steps": bool(next_steps) or "limitation" in text or "next" in text or "dry-run" in text,
    }
    format_score = sum(1 for ok in checks.values() if ok) / len(checks)
    return {
        "pass": checks["has_summary"] and checks["has_status"] and checks["has_tool_evidence"],
        "format_score": format_score,
        "checks": checks,
    }
