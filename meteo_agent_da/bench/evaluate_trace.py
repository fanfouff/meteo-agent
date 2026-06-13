from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pasbench import load_jsonl, score_required_tools


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MeteoAgent-DA trace against PASBench-DA tasks.")
    parser.add_argument("--tasks", default="examples/pasbench_da_sample.jsonl")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    used_tools = [item["name"] for item in report.get("tool_results", [])]
    request = report.get("request", "")

    matched = [task for task in tasks if task.prompt == request]
    if not matched:
        print(json.dumps({"matched": False, "used_tools": used_tools}, indent=2))
        return

    task = matched[0]
    score = score_required_tools(task.required_tools, used_tools)
    print(
        json.dumps(
            {
                "matched": True,
                "task_id": task.task_id,
                "category": task.category,
                "used_tools": used_tools,
                **score,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
