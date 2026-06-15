from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def executable_workflow_reward(
    tool_results: Iterable[Mapping[str, object]],
    required_artifacts: Iterable[str] = (),
    scientific_checks: Mapping[str, bool] | None = None,
    report: Mapping[str, object] | None = None,
) -> Dict[str, float]:
    results = list(tool_results)
    if not results:
        return _round_breakdown(
            {
                "total_reward": 0.0,
                "tool_success_reward": 0.0,
                "artifact_reward": 0.0,
                "scientific_check_reward": 0.0,
                "format_reward": 0.0,
                "reproducibility_reward": 0.0,
                "hallucination_penalty": 0.0,
                "unsafe_command_penalty": 0.0,
            }
        )

    success_rate = sum(1 for item in results if item.get("status") == "ok") / len(results)
    artifacts = {str(artifact) for item in results for artifact in item.get("artifacts", []) or []}
    if report:
        artifacts.update(str(item) for item in report.get("artifacts", []) or [])
    required = set(required_artifacts)
    artifact_score = 1.0 if not required else len(required & artifacts) / len(required)
    checks = scientific_checks or {}
    scientific_score = 1.0 if not checks else sum(1 for ok in checks.values() if ok) / len(checks)
    format_score = _format_reward(report, results)
    reproducibility_score = _reproducibility_reward(results)
    hallucination_penalty = _hallucination_penalty(artifacts)
    unsafe_command_penalty = _unsafe_command_penalty(results)

    total = (
        0.35 * success_rate
        + 0.20 * artifact_score
        + 0.20 * scientific_score
        + 0.10 * format_score
        + 0.15 * reproducibility_score
        + hallucination_penalty
        + unsafe_command_penalty
    )
    return _round_breakdown(
        {
            "total_reward": _clamp(total),
            "tool_success_reward": success_rate,
            "artifact_reward": artifact_score,
            "scientific_check_reward": scientific_score,
            "format_reward": format_score,
            "reproducibility_reward": reproducibility_score,
            "hallucination_penalty": hallucination_penalty,
            "unsafe_command_penalty": unsafe_command_penalty,
        }
    )


def pasbench_reward_breakdown(score_row: Mapping[str, Any]) -> Dict[str, float]:
    verifier_reward = _safe_float(score_row.get("verifier_pass", 0.0))
    tool_recall_reward = _safe_float(score_row.get("tool_recall", 0.0))
    command_reward = _safe_float(score_row.get("command_validity", 0.0))
    artifact_reward = _safe_float(score_row.get("artifact_recall", 0.0))
    hallucination_rate = _safe_float(score_row.get("hallucination_rate", 0.0))
    cost_reward = _safe_float(score_row.get("cost_score", 1.0))
    hallucination_penalty = -0.10 * _clamp(hallucination_rate)
    total = (
        0.30 * verifier_reward
        + 0.20 * tool_recall_reward
        + 0.15 * command_reward
        + 0.15 * artifact_reward
        + 0.10 * (1.0 - _clamp(hallucination_rate))
        + 0.10 * cost_reward
    )
    return _round_breakdown(
        {
            "total_reward": _clamp(total),
            "verifier_reward": verifier_reward,
            "tool_recall_reward": tool_recall_reward,
            "command_reward": command_reward,
            "artifact_reward": artifact_reward,
            "hallucination_penalty": hallucination_penalty,
            "cost_reward": cost_reward,
            "task_success_reward": _safe_float(score_row.get("pass_rate", 0.0)),
        }
    )


def _format_reward(report: Mapping[str, object] | None, results: list[Mapping[str, object]]) -> float:
    if not report:
        return 1.0
    checks = [
        bool(str(report.get("summary", "")).strip()),
        bool(results),
        bool(report.get("next_steps", [])),
        str(report.get("status", "")) in {"ok", "error", "skipped"},
    ]
    return sum(1 for item in checks if item) / len(checks)


def _reproducibility_reward(results: list[Mapping[str, object]]) -> float:
    command_results = [item for item in results if item.get("name") == "pasnet_runner"]
    if not command_results:
        return 1.0
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
    scores = []
    for result in command_results:
        data = result.get("data", {}) or {}
        if not isinstance(data, Mapping):
            scores.append(0.0)
            continue
        commands = data.get("commands", []) or []
        if not commands:
            scores.append(0.0)
            continue
        for command in commands:
            if not isinstance(command, Mapping):
                scores.append(0.0)
                continue
            args = [str(item) for item in command.get("command", []) or []]
            if not args:
                scores.append(0.0)
                continue
            split_file = _value_after(args, "--split_file")
            split_ok = bool(split_file) and (not Path(split_file).is_absolute() or Path(split_file).exists())
            args_ok = required_args.issubset(set(args))
            scores.append(1.0 if args_ok and split_ok else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _hallucination_penalty(artifacts: Iterable[str]) -> float:
    paths = [str(item) for item in artifacts if str(item)]
    if not paths:
        return 0.0
    missing = sum(1 for item in paths if not Path(item).exists())
    return -0.20 * (missing / len(paths))


def _unsafe_command_penalty(results: list[Mapping[str, object]]) -> float:
    unsafe_tokens = {"rm", "sudo", "mkfs", "dd", "shutdown", "reboot"}
    unsafe = 0
    total = 0
    for result in results:
        data = result.get("data", {}) or {}
        if not isinstance(data, Mapping):
            continue
        for command in data.get("commands", []) or []:
            if not isinstance(command, Mapping):
                continue
            args = [Path(str(item)).name for item in command.get("command", [])[:3]]
            total += 1
            if set(args).intersection(unsafe_tokens):
                unsafe += 1
    if total == 0:
        return 0.0
    return -0.30 * (unsafe / total)


def _value_after(args: list[str], key: str) -> str:
    try:
        index = args.index(key)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    value = args[index + 1]
    return "" if value.startswith("--") else value


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round_breakdown(payload: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 4) for key, value in payload.items()}
