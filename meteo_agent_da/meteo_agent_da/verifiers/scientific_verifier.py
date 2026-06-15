from __future__ import annotations

import re
from typing import Any, Dict, Optional


FALSE_TRAINING_PATTERNS = [
    r"\btraining completed\b",
    r"\bsuccessfully trained\b",
    r"\bfinished training\b",
    r"\bmodel was trained\b",
    r"训练已完成",
    r"已经完成训练",
    r"已完成真实训练",
]

METRIC_CLAIM_PATTERN = re.compile(r"\b(RMSE|MAE|ACC|bias)\b[^0-9]{0,16}[0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def verify_scientific_consistency(
    report: Dict[str, Any],
    dry_run_expected: Optional[bool] = None,
) -> Dict[str, Any]:
    text = _report_text(report)
    lower = text.lower()
    pasnet_used = any(
        str(item.get("name", "")) == "pasnet_runner"
        for item in report.get("tool_results", []) or []
    )
    evaluator_used = any(
        str(item.get("name", "")) == "evaluator"
        for item in report.get("tool_results", []) or []
    )

    false_training_claim = any(re.search(pattern, lower, re.IGNORECASE) for pattern in FALSE_TRAINING_PATTERNS)
    invented_metric_claim = bool(METRIC_CLAIM_PATTERN.search(text)) and not evaluator_used
    dry_run_disclosed = True
    if dry_run_expected is True and pasnet_used:
        dry_run_disclosed = any(
            marker in lower
            for marker in ("dry_run=true", "dry-run", "not executed", "not execute", "commands are not executed")
        )

    checks = {
        "no_false_training_claim": not (dry_run_expected is True and false_training_claim),
        "no_ungrounded_metric_claim": not invented_metric_claim,
        "dry_run_disclosed": dry_run_disclosed,
    }
    scientific_score = sum(1 for ok in checks.values() if ok) / len(checks)
    return {
        "pass": all(checks.values()),
        "scientific_score": scientific_score,
        "checks": checks,
        "false_training_claim": false_training_claim,
        "invented_metric_claim": invented_metric_claim,
    }


def _report_text(report: Dict[str, Any]) -> str:
    parts = [
        str(report.get("request", "")),
        str(report.get("summary", "")),
        " ".join(str(item) for item in report.get("next_steps", []) or []),
    ]
    for result in report.get("tool_results", []) or []:
        parts.append(str(result.get("summary", "")))
        parts.append(str(result.get("data", "")))
        parts.append(str(result.get("error", "")))
    return "\n".join(parts)
