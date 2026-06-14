from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..domains import get_profile


def verify_report(task: Any, report: Dict[str, Any]) -> Dict[str, Any]:
    """Run executable checks over a report.

    These checks intentionally inspect tool observations rather than free-form
    text. That keeps the benchmark grounded in paths, commands, metrics,
    artifacts, and cost signals that can be audited after a run.
    """

    tool_results = list(report.get("tool_results", []))
    used_tools = [str(item.get("name", "")) for item in tool_results]
    checks = _selected_checks(task)

    path_validity = _verify_paths(task, tool_results, used_tools)
    command_validity = _verify_commands(task, tool_results, used_tools)
    metric_validity = _verify_metrics(task, tool_results, used_tools)
    artifact_validity = _verify_artifacts(task, report, tool_results, used_tools)
    cost_score = _verify_cost(task, used_tools)
    hallucination_rate = _artifact_hallucination_rate(report.get("artifacts", []))

    values = {
        "path": path_validity,
        "command": command_validity,
        "metric": metric_validity,
        "artifact": artifact_validity,
        "cost": 1.0 if cost_score >= 1.0 else 0.0,
    }
    failed_checks = [name for name in checks if values.get(name, 1.0) < 1.0]
    report_ok = report.get("status") == "ok"
    verifier_pass = 1.0 if report_ok and not failed_checks and hallucination_rate == 0.0 else 0.0

    return {
        "verifier_pass": verifier_pass,
        "path_validity": path_validity,
        "command_validity": command_validity,
        "metric_validity": metric_validity,
        "artifact_validity": artifact_validity,
        "cost_score": cost_score,
        "hallucination_rate": hallucination_rate,
        "failed_checks": failed_checks,
    }


def _selected_checks(task: Any) -> List[str]:
    explicit = list(getattr(task, "verifier_checks", None) or [])
    if explicit:
        return explicit

    rubric = dict(getattr(task, "rubric", {}) or {})
    checks = []
    if any(key in rubric for key in ("path_validity", "split_count_check")):
        checks.append("path")
    if "command_validity" in rubric:
        checks.append("command")
    if any(key in rubric for key in ("metric_grounding", "metric_validity")):
        checks.append("metric")
    if any(key in rubric for key in ("paper_artifact", "figure_evidence", "artifact_validity")):
        checks.append("artifact")

    profile = get_profile(getattr(task, "domain_profile", "pasnet_satellite"))
    for item in profile.verifier_checks:
        if item not in checks:
            checks.append(item)
    return checks or ["path", "command", "metric", "artifact", "cost"]


def _verify_paths(task: Any, tool_results: List[Dict[str, Any]], used_tools: List[str]) -> float:
    required = set(getattr(task, "required_tools", []) or [])
    needs_data = "data_indexer" in required or "data_indexer" in used_tools
    needs_sanity = "sanity_check" in required or "sanity_check" in used_tools
    if not needs_data and not needs_sanity:
        return 1.0

    scores = []
    sanity = _find_result(tool_results, "sanity_check")
    if needs_sanity:
        scores.append(1.0 if sanity and sanity.get("status") == "ok" else 0.0)
        if sanity:
            checks = dict(sanity.get("data", {}).get("checks", {}) or {})
            exists_checks = [value for key, value in checks.items() if key.endswith("_exists")]
            if exists_checks:
                scores.append(sum(1 for value in exists_checks if value) / len(exists_checks))

    indexer = _find_result(tool_results, "data_indexer")
    if needs_data:
        scores.append(1.0 if indexer and indexer.get("status") == "ok" else 0.0)
        if indexer:
            data = dict(indexer.get("data", {}) or {})
            path_keys = ["data_root_exists", "split_file_exists", "stats_file_exists", "increment_stats_exists"]
            checks = [bool(data.get(key)) for key in path_keys if key in data]
            if checks:
                scores.append(sum(1 for value in checks if value) / len(checks))
            split_counts = dict(data.get("split_counts", {}) or {})
            if split_counts:
                scores.append(1.0 if all(int(value) >= 0 for value in split_counts.values()) else 0.0)

    return min(scores) if scores else 1.0


def _verify_commands(task: Any, tool_results: List[Dict[str, Any]], used_tools: List[str]) -> float:
    required = set(getattr(task, "required_tools", []) or [])
    if "pasnet_runner" not in required and "pasnet_runner" not in used_tools:
        return 1.0

    runner = _find_result(tool_results, "pasnet_runner")
    if not runner or runner.get("status") != "ok":
        return 0.0
    data = dict(runner.get("data", {}) or {})
    if data.get("split_file_exists") is False:
        return 0.0

    commands = list(data.get("commands", []) or [])
    if not commands:
        return 0.0

    required_args = {
        "--exp_name",
        "--output_dir",
        "--data_root",
        "--stats_file",
        "--increment_stats",
        "--split_mode",
        "--split_file",
        "--model",
        "--epochs",
        "--batch_size",
        "--lr",
        "--loss",
    }
    allowed_models = {"pasnet", "physics_unet", "swin_unet", "fuxi_da", "mamba", "vanilla_unet"}
    valid = 0
    for command in commands:
        args = [str(item) for item in command.get("command", [])]
        model = str(command.get("model", ""))
        has_required_args = required_args.issubset(set(args))
        split_file = _value_after(args, "--split_file")
        split_ok = bool(split_file and Path(split_file).exists())
        model_ok = model in allowed_models or bool(model)
        if has_required_args and split_ok and model_ok:
            valid += 1
    return valid / len(commands)


def _verify_metrics(task: Any, tool_results: List[Dict[str, Any]], used_tools: List[str]) -> float:
    required = set(getattr(task, "required_tools", []) or [])
    if "evaluator" in required and "evaluator" not in used_tools:
        return 0.0
    if "evaluator" not in used_tools:
        return 1.0
    evaluator = _find_result(tool_results, "evaluator")
    if not evaluator or evaluator.get("status") != "ok":
        return 0.0
    data = dict(evaluator.get("data", {}) or {})
    if "metric_files" not in data:
        return 0.0
    summaries = list(data.get("summaries", []) or [])
    if not summaries:
        return 1.0
    parsed = sum(1 for item in summaries if "error" not in item)
    return parsed / len(summaries)


def _verify_artifacts(
    task: Any,
    report: Dict[str, Any],
    tool_results: List[Dict[str, Any]],
    used_tools: List[str],
) -> float:
    expected = list(getattr(task, "expected_artifacts", []) or [])
    artifacts = [str(item) for item in report.get("artifacts", [])]
    if expected:
        matches = sum(1 for item in expected if any(item in artifact for artifact in artifacts))
        return matches / len(expected)

    artifact_tools = {"paper_writer", "plotter", "evaluator"}
    required = set(getattr(task, "required_tools", []) or [])
    if artifact_tools.intersection(required) and not artifact_tools.intersection(used_tools):
        return 0.0
    if not artifact_tools.intersection(used_tools):
        return 1.0

    produced = []
    for result in tool_results:
        if result.get("name") in artifact_tools:
            produced.extend(str(item) for item in result.get("artifacts", []))
    if not produced:
        return 0.0 if "paper_writer" in used_tools else 1.0

    existing = sum(1 for item in produced if Path(item).exists())
    return existing / len(produced)


def _verify_cost(task: Any, used_tools: List[str]) -> float:
    max_steps = int(getattr(task, "max_tool_steps", 8) or 8)
    steps = len(used_tools)
    if steps <= max_steps:
        return 1.0
    return max_steps / steps if steps else 1.0


def _artifact_hallucination_rate(artifacts: Iterable[Any]) -> float:
    paths = [str(item) for item in artifacts]
    if not paths:
        return 0.0
    missing = sum(1 for item in paths if not Path(item).exists())
    return missing / len(paths)


def _find_result(tool_results: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for result in tool_results:
        if result.get("name") == name:
            return result
    return None


def _value_after(args: List[str], key: str) -> str:
    try:
        index = args.index(key)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    return args[index + 1]
