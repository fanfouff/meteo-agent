from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BenchTask:
    task_id: str
    category: str
    prompt: str
    required_tools: List[str]
    verifier: str
    expected_artifacts: List[str]
    max_tool_steps: int
    rubric: Dict[str, Any]
    domain_profile: str = "pasnet_satellite"
    gold_tool_path: Optional[List[str]] = None
    allowed_alternatives: Optional[Dict[str, List[str]]] = None
    verifier_checks: Optional[List[str]] = None


def load_jsonl(path: Path) -> list[BenchTask]:
    tasks: list[BenchTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            tasks.append(
                BenchTask(
                    task_id=raw["task_id"],
                    category=raw["category"],
                    prompt=raw["prompt"],
                    required_tools=list(raw.get("required_tools", [])),
                    verifier=raw.get("verifier", "manual"),
                    expected_artifacts=list(raw.get("expected_artifacts", [])),
                    max_tool_steps=int(raw.get("max_tool_steps", 8)),
                    rubric=dict(raw.get("rubric", {})),
                    domain_profile=raw.get("domain_profile", "pasnet_satellite"),
                    gold_tool_path=list(raw.get("gold_tool_path", raw.get("required_tools", []))),
                    allowed_alternatives=dict(raw.get("allowed_alternatives", {})),
                    verifier_checks=list(raw.get("verifier_checks", [])),
                )
            )
    return tasks


def score_required_tools(required_tools: Iterable[str], used_tools: Iterable[str]) -> dict:
    required = set(required_tools)
    used = set(used_tools)
    if not required:
        return {"tool_recall": 1.0, "missing_tools": []}
    missing = sorted(required - used)
    return {"tool_recall": (len(required) - len(missing)) / len(required), "missing_tools": missing}


def score_tool_precision(required_tools: Iterable[str], used_tools: Iterable[str]) -> dict:
    required = set(required_tools)
    used = list(used_tools)
    if not used:
        return {"tool_precision": 1.0 if not required else 0.0, "unexpected_tools": []}
    unexpected = sorted(set(used) - required) if required else []
    correct = sum(1 for item in used if not required or item in required)
    return {"tool_precision": correct / len(used), "unexpected_tools": unexpected}


def score_expected_artifacts(expected_artifacts: Iterable[str], artifacts: Iterable[str]) -> dict:
    expected = [str(item) for item in expected_artifacts]
    actual = [str(item) for item in artifacts]
    if not expected:
        return {"artifact_recall": 1.0, "missing_artifacts": []}
    missing = sorted(item for item in expected if not any(item in artifact for artifact in actual))
    return {"artifact_recall": (len(expected) - len(missing)) / len(expected), "missing_artifacts": missing}


def score_report(task: BenchTask, report: Dict[str, Any]) -> Dict[str, Any]:
    from .verifiers import verify_report

    tool_results = list(report.get("tool_results", []))
    used_tools = [item.get("name", "") for item in tool_results]
    artifacts = list(report.get("artifacts", []))
    tool_steps = len(used_tools)
    ok_tools = sum(1 for item in tool_results if item.get("status") == "ok")
    repeated_tools = max(0, tool_steps - len(set(used_tools)))

    required_score = score_required_tools(task.required_tools, used_tools)
    precision_score = score_tool_precision(task.required_tools, used_tools)
    artifact_score = score_expected_artifacts(task.expected_artifacts, artifacts)
    verifier_score = verify_report(task, report)
    tool_success_rate = ok_tools / tool_steps if tool_steps else 0.0
    budget_ok = tool_steps <= task.max_tool_steps
    status_ok = report.get("status") == "ok"
    pass_rate = 1.0 if (
        status_ok
        and required_score["tool_recall"] == 1.0
        and artifact_score["artifact_recall"] == 1.0
        and budget_ok
        and tool_success_rate == 1.0
        and verifier_score["verifier_pass"] == 1.0
    ) else 0.0

    return {
        "task_id": task.task_id,
        "category": task.category,
        "domain_profile": task.domain_profile,
        "status_ok": status_ok,
        "used_tools": used_tools,
        "tool_steps": tool_steps,
        "tool_success_rate": tool_success_rate,
        "repeated_tool_calls": repeated_tools,
        "budget_ok": budget_ok,
        "artifact_count": len(artifacts),
        "pass_rate": pass_rate,
        **required_score,
        **precision_score,
        **artifact_score,
        **verifier_score,
    }


def aggregate_scores(scores: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(scores)
    if not rows:
        return {"num_tasks": 0}
    numeric_keys = [
        "tool_recall",
        "tool_precision",
        "tool_success_rate",
        "artifact_recall",
        "pass_rate",
        "verifier_pass",
        "command_validity",
        "artifact_validity",
        "hallucination_rate",
        "cost_score",
        "repeated_tool_calls",
        "tool_steps",
    ]
    output: Dict[str, Any] = {"num_tasks": len(rows)}
    for key in numeric_keys:
        output[f"avg_{key}"] = sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    output["budget_ok_rate"] = sum(1 for row in rows if row.get("budget_ok")) / len(rows)
    output["status_ok_rate"] = sum(1 for row in rows if row.get("status_ok")) / len(rows)
    return output
