from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .artifact_verifier import verify_artifacts
from .command_verifier import verify_commands
from .metric_verifier import verify_metrics
from .report_verifier import verify_report_format
from .scientific_verifier import verify_scientific_consistency


def verify_report_quality(
    report: Dict[str, Any],
    expected_artifacts: Iterable[str] = (),
    required_tools: Iterable[str] = (),
    max_tool_steps: Optional[int] = None,
    dry_run_expected: Optional[bool] = None,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run post-training quality gates over one agent report.

    This verifier is intentionally independent from PASBench scoring. PASBench
    decides whether a task is solved; this function decides whether a rollout is
    clean enough to become SFT, DPO, or reward-model data.
    """

    tool_results = list(report.get("tool_results", []) or [])
    used_tools = [str(item.get("name", "")) for item in tool_results]
    required = [str(item) for item in required_tools]
    missing_tools = sorted(set(required) - set(used_tools))
    failed_tools = [
        str(item.get("name", ""))
        for item in tool_results
        if str(item.get("status", "")) not in {"ok", "skipped"}
    ]
    tool_evidence = bool(tool_results)
    tool_budget_ok = True if max_tool_steps is None else len(tool_results) <= max_tool_steps

    artifact = verify_artifacts(report, expected_artifacts=expected_artifacts, base_dir=base_dir)
    command = verify_commands(report)
    metric = verify_metrics(report)
    report_format = verify_report_format(report)
    scientific = verify_scientific_consistency(report, dry_run_expected=dry_run_expected)

    checks = {
        "status_ok": report.get("status") == "ok",
        "tool_evidence": tool_evidence,
        "required_tools": not missing_tools,
        "tool_success": not failed_tools,
        "tool_budget": tool_budget_ok,
        "artifact": bool(artifact["pass"]),
        "command": bool(command["pass"]),
        "metric": bool(metric["pass"]),
        "report_format": bool(report_format["pass"]),
        "scientific": bool(scientific["pass"]),
    }
    failed_checks = [name for name, ok in checks.items() if not ok]

    return {
        "verifier_pass": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "missing_tools": missing_tools,
        "failed_tools": failed_tools,
        "used_tools": used_tools,
        "artifact": artifact,
        "command": command,
        "metric": metric,
        "report_format": report_format,
        "scientific": scientific,
    }


__all__ = [
    "verify_artifacts",
    "verify_commands",
    "verify_metrics",
    "verify_report_format",
    "verify_scientific_consistency",
    "verify_report_quality",
]
