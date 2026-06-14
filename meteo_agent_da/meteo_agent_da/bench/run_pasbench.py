from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..agent.llm_planner import QwenPlanner
from ..agent.runtime import MeteoAgentRuntime
from ..agent.schemas import AgentReport, AgentTask, ProjectConfig, ToolResult, ToolStatus
from ..baselines.text_only import build_text_only_report
from ..tools.registry import build_default_registry
from .pasbench import aggregate_scores, load_jsonl, score_report


METHODS = ("text_only", "pico", "heuristic_tool", "qwen_tool", "sft_qwen")


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    tasks = load_jsonl(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    reports_by_method: Dict[str, List[str]] = {method: [] for method in args.method}
    successful_reports: List[str] = []

    for method in args.method:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            for attempt in range(1, args.attempts + 1):
                report, report_path = _run_one(method, task, attempt, method_dir, args)
                reports_by_method[method].append(str(report_path))
                score = score_report(task, report.model_dump())
                row = {
                    "method": method,
                    "attempt": attempt,
                    "report": str(report_path),
                    **score,
                }
                all_rows.append(row)
                if score["pass_rate"] == 1.0:
                    successful_reports.append(str(report_path))

    aggregate = {
        method: aggregate_scores(row for row in all_rows if row["method"] == method)
        for method in args.method
    }
    payload = {
        "tasks": str(Path(args.tasks)),
        "methods": args.method,
        "attempts": args.attempts,
        "scores": all_rows,
        "aggregate": aggregate,
        "reports_by_method": reports_by_method,
        "successful_reports": successful_reports,
    }

    (output_dir / "scores.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_scores_csv(output_dir / "scores.csv", all_rows)
    _write_core_results(output_dir / "core_results.md", aggregate)
    _write_core_results_csv(output_dir / "core_results.csv", aggregate)
    (output_dir / "successful_reports.txt").write_text("\n".join(successful_reports) + ("\n" if successful_reports else ""), encoding="utf-8")
    return payload


def _run_one(method: str, task: Any, attempt: int, method_dir: Path, args: argparse.Namespace) -> Tuple[AgentReport, Path]:
    run_id = f"{task.task_id}_{method}_attempt{attempt}"
    task_run_dir = method_dir / run_id
    task_run_dir.mkdir(parents=True, exist_ok=True)
    agent_task = AgentTask(request=task.prompt, run_id=run_id, dry_run=args.dry_run, max_steps=task.max_tool_steps)

    if method == "text_only":
        report = build_text_only_report(agent_task)
        report_path = _write_report(task_run_dir, report)
        return report, report_path

    if method == "pico":
        report = _run_pico(agent_task, task_run_dir, args.pico_command)
        report_path = _write_report(task_run_dir, report)
        return report, report_path

    config = ProjectConfig(project_root=Path(args.project_root), dry_run=args.dry_run)
    registry = build_default_registry()
    planner = None
    if method in {"qwen_tool", "sft_qwen"}:
        planner = QwenPlanner.from_env(
            tool_specs=registry.specs(),
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
        )
    runtime = MeteoAgentRuntime(config=config, registry=registry, planner=planner, run_root=method_dir)
    report = runtime.run(agent_task)
    return report, task_run_dir / "report.json"


def _run_pico(task: AgentTask, run_dir: Path, pico_command: str) -> AgentReport:
    command = pico_command or os.getenv("PICO_COMMAND", "")
    if not command:
        return AgentReport(
            run_id=task.run_id,
            request=task.request,
            status=ToolStatus.SKIPPED,
            summary="Pico baseline skipped because --pico-command or PICO_COMMAND is not configured.",
            tool_results=[
                ToolResult(
                    name="pico_cli",
                    status=ToolStatus.SKIPPED,
                    summary="External Pico command was not configured; no Pico code is vendored into MeteoAgent-DA.",
                )
            ],
            artifacts=[],
            next_steps=["Set --pico-command with a local Pico CLI wrapper to enable this baseline."],
        )

    output_path = run_dir / "pico_stdout.txt"
    stderr_path = run_dir / "pico_stderr.txt"
    formatted = command.format(task=shlex.quote(task.request), output=shlex.quote(str(output_path)))
    cmd = shlex.split(formatted)
    if "{task}" not in command:
        cmd.append(task.request)

    try:
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except Exception as exc:
        return AgentReport(
            run_id=task.run_id,
            request=task.request,
            status=ToolStatus.ERROR,
            summary=f"Pico baseline failed to start: {type(exc).__name__}: {exc}",
            tool_results=[
                ToolResult(name="pico_cli", status=ToolStatus.ERROR, summary="Pico process failed to start.", error=str(exc))
            ],
        )

    output_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    status = ToolStatus.OK if completed.returncode == 0 else ToolStatus.ERROR
    return AgentReport(
        run_id=task.run_id,
        request=task.request,
        status=status,
        summary=f"Pico command exited with code {completed.returncode}. Output saved under {run_dir}.",
        tool_results=[
            ToolResult(
                name="pico_cli",
                status=status,
                summary=f"External Pico command returncode={completed.returncode}.",
                data={"stdout": str(output_path), "stderr": str(stderr_path), "command": cmd},
                artifacts=[str(output_path), str(stderr_path)],
                error=None if status == ToolStatus.OK else "pico_command_failed",
            )
        ],
        artifacts=[str(output_path), str(stderr_path)],
    )


def _write_report(run_dir: Path, report: AgentReport) -> Path:
    path = run_dir / "report.json"
    path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_scores_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (list, dict))})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_core_results(path: Path, aggregate: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "| Method | TSR | VER | TCR | CVR | AGR | HAL↓ | Cost↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, scores in aggregate.items():
        lines.append(
            "| {method} | {tsr:.3f} | {ver:.3f} | {tcr:.3f} | {cvr:.3f} | {agr:.3f} | {hal:.3f} | {cost:.3f} |".format(
                method=method,
                tsr=float(scores.get("avg_pass_rate", 0.0)),
                ver=float(scores.get("avg_verifier_pass", 0.0)),
                tcr=float(scores.get("avg_tool_recall", 0.0)),
                cvr=float(scores.get("avg_command_validity", 0.0)),
                agr=float(scores.get("avg_artifact_validity", 0.0)),
                hal=float(scores.get("avg_hallucination_rate", 0.0)),
                cost=float(scores.get("avg_tool_steps", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_core_results_csv(path: Path, aggregate: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for method, scores in aggregate.items():
        rows.append(
            {
                "method": method,
                "TSR": float(scores.get("avg_pass_rate", 0.0)),
                "VER": float(scores.get("avg_verifier_pass", 0.0)),
                "TCR": float(scores.get("avg_tool_recall", 0.0)),
                "CVR": float(scores.get("avg_command_validity", 0.0)),
                "AGR": float(scores.get("avg_artifact_validity", 0.0)),
                "HAL": float(scores.get("avg_hallucination_rate", 0.0)),
                "Cost": float(scores.get("avg_tool_steps", 0.0)),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "TSR", "VER", "TCR", "CVR", "AGR", "HAL", "Cost"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PASBench-DA across baselines and tool agents.")
    parser.add_argument("--tasks", default="examples/pasbench_da_50.jsonl")
    parser.add_argument("--method", nargs="+", choices=METHODS, default=["text_only", "heuristic_tool"])
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/pasbench")
    parser.add_argument("--project-root", default="/home/lrx/Unet/satellite_assimilation_v2")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--pico-command", default="", help="External Pico CLI command. Supports {task} and {output} placeholders.")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key-env", default="QWEN_API_KEY")
    args = parser.parse_args()
    args.dry_run = False if args.execute else True
    return args


def main() -> None:
    payload = run_benchmark(parse_args())
    print(json.dumps({"aggregate": payload["aggregate"], "successful_reports": len(payload["successful_reports"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
