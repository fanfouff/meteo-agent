from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pasbench import aggregate_scores, load_jsonl, score_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MeteoAgent-DA trace against PASBench-DA tasks.")
    parser.add_argument("--tasks", default="examples/pasbench_da_sample.jsonl")
    parser.add_argument("--report", required=True, nargs="+")
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    scores = []
    for item in args.report:
        report = json.loads(Path(item).read_text(encoding="utf-8"))
        request = report.get("request", "")
        matched = [task for task in tasks if task.prompt == request or request.endswith(task.prompt)]
        if not matched:
            scores.append(
                {
                    "matched": False,
                    "report": item,
                    "request": request,
                    "used_tools": [result["name"] for result in report.get("tool_results", [])],
                }
            )
            continue
        score = score_report(matched[0], report)
        scores.append({"matched": True, "report": item, **score})

    print(json.dumps({"scores": scores, "aggregate": aggregate_scores([row for row in scores if row.get("matched")])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
