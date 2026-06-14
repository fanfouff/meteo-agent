from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .memory import WorkingMemory
from .planner import HeuristicPlanner, Planner
from .schemas import AgentReport, AgentTask, ProjectConfig, StepKind, ToolResult, ToolStatus, TraceEvent
from .tool_registry import ToolRegistry
from ..tools.registry import build_default_registry


class MeteoAgentRuntime:
    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        registry: Optional[ToolRegistry] = None,
        planner: Optional[Planner] = None,
        run_root: Optional[Path] = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.registry = registry or build_default_registry()
        self.planner = planner or HeuristicPlanner()
        self.run_root = run_root or Path.cwd() / "runs"

    def run(self, task: AgentTask) -> AgentReport:
        self.config.dry_run = task.dry_run
        run_dir = self.run_root / task.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        memory = WorkingMemory(task_summary=task.request)
        trace_path = run_dir / "trace.jsonl"

        def trace(kind: StepKind, message: str, payload: dict) -> None:
            step = 0
            if trace_path.exists():
                with trace_path.open("r", encoding="utf-8") as existing:
                    step = sum(1 for _ in existing)
            event = TraceEvent(
                run_id=task.run_id,
                step=step,
                kind=kind,
                message=message,
                payload=payload,
            )
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")

        plan = self.planner.plan(task)
        trace(StepKind.PLAN, "Plan created", plan.model_dump())

        results: list[ToolResult] = []
        for index, call in enumerate(plan.tool_calls[: task.max_steps], start=1):
            trace(StepKind.TOOL, f"Running tool {call.name}", call.model_dump())
            result = self.registry.run(call, self.config)
            results.append(result)
            memory.add_tool_result(result)
            trace(StepKind.REFLECT, f"Observed {call.name}: {result.status}", result.model_dump())
            if result.status == ToolStatus.ERROR:
                memory.add_observation(f"Tool {call.name} failed: {result.error}")

        status = ToolStatus.OK if all(r.status != ToolStatus.ERROR for r in results) else ToolStatus.ERROR
        artifacts = [item for result in results for item in result.artifacts]
        summary = self._build_summary(task, results, memory)

        report = AgentReport(
            run_id=task.run_id,
            request=task.request,
            status=status,
            summary=summary,
            tool_results=results,
            artifacts=artifacts,
            next_steps=[
                "Replace the heuristic planner with an LLM planner once the tool contracts are stable.",
                "Add PASBench-DA tasks for this workflow and convert successful traces into SFT samples.",
            ],
        )

        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        trace(StepKind.REPORT, "Report written", {"report_path": str(report_path), "memory": memory.snapshot()})
        return report

    @staticmethod
    def _build_summary(task: AgentTask, results: list[ToolResult], memory: WorkingMemory) -> str:
        ok = sum(1 for result in results if result.status == ToolStatus.OK)
        err = sum(1 for result in results if result.status == ToolStatus.ERROR)
        lines = [
            f"Request: {task.request}",
            f"Tool calls completed: {ok} ok, {err} error.",
        ]
        lines.extend(f"- {result.name}: {result.summary}" for result in results)
        return "\n".join(lines)
