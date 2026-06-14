from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..agent.schemas import AgentReport, AgentTask, ToolStatus


def build_text_only_report(task: AgentTask) -> AgentReport:
    summary = (
        "Text-only baseline: this answer does not call PASNet-DA tools. "
        "It can outline a plausible research plan, but it cannot verify paths, "
        "metrics, commands, or artifacts."
    )
    return AgentReport(
        run_id=task.run_id,
        request=task.request,
        status=ToolStatus.OK,
        summary=summary,
        tool_results=[],
        artifacts=[],
        next_steps=[
            "Compare this baseline against a tool-agent report using PASBench-DA.",
            "Use tool evidence before claiming experiment validity.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-tool text baseline for PASBench-DA.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = build_text_only_report(AgentTask(request=args.task, dry_run=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(output)
    else:
        print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
