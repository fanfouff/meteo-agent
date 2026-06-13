from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable


def verify_report(report: Dict[str, Any], required: Iterable[str] = ()) -> Dict[str, Any]:
    required_checks = set(required)
    checks = {
        "path": verify_paths(report),
        "command": verify_commands(report),
        "metric": verify_metrics(report),
        "artifact": verify_artifacts(report),
    }
    selected = {key: value for key, value in checks.items() if not required_checks or key in required_checks}
    passed = sum(1 for value in selected.values() if value["ok"])
    total = len(selected)
    return {
        "verifier_pass_rate": passed / total if total else 1.0,
        "verifier_results": checks,
    }


def verify_paths(report: Dict[str, Any]) -> Dict[str, Any]:
    data = _tool_data(report, "data_indexer")
    if not data:
        data = _tool_data(report, "sanity_check").get("checks", {})
    keys = [
        "data_root_exists",
        "split_file_exists",
        "stats_file_exists",
        "increment_stats_exists",
        "project_root_exists",
        "train_script_exists",
        "default_data_root_exists",
        "default_stats_file_exists",
        "default_increment_stats_exists",
        "default_split_dir_exists",
    ]
    seen = {key: bool(data[key]) for key in keys if key in data}
    if not seen:
        return {"ok": True, "details": "no path-sensitive tool output"}
    missing = [key for key, ok in seen.items() if not ok]
    return {"ok": not missing, "missing": missing}


def verify_commands(report: Dict[str, Any]) -> Dict[str, Any]:
    data = _tool_data(report, "pasnet_runner")
    commands = data.get("commands", [])
    if not commands:
        return {"ok": True, "details": "no command-generation tool output"}
    missing = []
    for item in commands:
        command = list(item.get("command", []))
        if "--model" not in command or "--split_file" not in command or "--data_root" not in command:
            missing.append(item.get("exp_name") or item.get("model") or "unknown")
    split_ok = bool(data.get("split_file_exists", True))
    return {"ok": not missing and split_ok, "invalid_commands": missing, "split_file_exists": split_ok}


def verify_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    data = _tool_data(report, "evaluator")
    summaries = data.get("summaries", [])
    if not summaries:
        return {"ok": True, "details": "no metric parser output"}
    parseable = 0
    for item in summaries:
        values = item.get("values", {})
        rows = item.get("rows_preview", [])
        if values or rows:
            parseable += 1
    return {"ok": parseable > 0, "parseable_summaries": parseable, "total_summaries": len(summaries)}


def verify_artifacts(report: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = list(report.get("artifacts", []))
    if not artifacts:
        return {"ok": True, "details": "no artifact required"}
    existing = [item for item in artifacts if Path(item).exists()]
    return {"ok": len(existing) == len(artifacts), "existing": existing, "total": len(artifacts)}


def _tool_data(report: Dict[str, Any], name: str) -> Dict[str, Any]:
    for item in report.get("tool_results", []):
        if item.get("name") == name:
            return dict(item.get("data", {}))
    return {}
