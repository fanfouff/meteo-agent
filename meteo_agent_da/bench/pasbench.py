from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class BenchTask:
    task_id: str
    category: str
    prompt: str
    required_tools: List[str]
    verifier: str


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
