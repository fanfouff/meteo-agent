from __future__ import annotations

from typing import Any, Dict


def verify_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    evaluator_results = [
        item for item in report.get("tool_results", []) or []
        if str(item.get("name", "")) == "evaluator"
    ]
    if not evaluator_results:
        return {
            "pass": True,
            "metric_validity": 1.0,
            "metric_result_count": 0,
            "parsed_summary_count": 0,
            "metric_errors": [],
        }

    metric_errors = []
    parsed = 0
    total = 0
    for result in evaluator_results:
        if result.get("status") != "ok":
            metric_errors.append({"tool_status": result.get("status"), "error": result.get("error")})
            continue
        data = dict(result.get("data", {}) or {})
        summaries = list(data.get("summaries", []) or [])
        if not summaries and "metric_files" not in data:
            metric_errors.append({"reason": "missing_metric_files"})
        for item in summaries:
            total += 1
            if isinstance(item, dict) and "error" not in item:
                parsed += 1
            else:
                metric_errors.append({"summary": item})

    metric_validity = 1.0 if total == 0 else parsed / total
    return {
        "pass": not metric_errors or metric_validity == 1.0,
        "metric_validity": metric_validity,
        "metric_result_count": len(evaluator_results),
        "parsed_summary_count": parsed,
        "metric_errors": metric_errors,
    }
