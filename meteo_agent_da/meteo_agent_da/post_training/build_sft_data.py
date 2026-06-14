from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def trace_to_sft(report_path: Path, trace_path: Optional[Path] = None) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trace_path = trace_path or report_path.with_name("trace.jsonl")
    trace_events = _load_trace(trace_path)
    trajectory = _trajectory_from_trace(trace_events, report)
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are MeteoAgent-DA, a satellite data-assimilation research agent. Use tools, verify paths, and report reproducible evidence.",
            },
            {"role": "user", "content": report["request"]},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "plan": trajectory["plan"],
                        "tool_trajectory": trajectory["tool_trajectory"],
                        "reflections": trajectory["reflections"],
                        "final_summary": report["summary"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "metadata": {
            "run_id": report["run_id"],
            "status": report["status"],
            "artifacts": report.get("artifacts", []),
            "report_path": str(report_path),
            "trace_path": str(trace_path) if trace_path.exists() else "",
            "trace_event_count": len(trace_events),
        },
    }


def _load_trace(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def _trajectory_from_trace(trace_events: List[Dict[str, Any]], report: Dict[str, Any]) -> Dict[str, Any]:
    plan: Dict[str, Any] = {}
    pending_calls: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    reflections: List[str] = []

    for event in trace_events:
        kind = event.get("kind")
        payload = event.get("payload", {})
        if kind == "plan":
            plan = payload
        elif kind == "tool":
            pending_calls.append(payload)
        elif kind == "reflect":
            observations.append(payload)
            message = event.get("message", "")
            if message:
                reflections.append(message)

    if not observations:
        observations = list(report.get("tool_results", []))

    tool_trajectory = []
    for index, observation in enumerate(observations):
        call = pending_calls[index] if index < len(pending_calls) else {}
        tool_trajectory.append(
            {
                "tool_call": {
                    "name": call.get("name", observation.get("name", "")),
                    "arguments": call.get("arguments", {}),
                    "reason": call.get("reason", ""),
                },
                "observation": {
                    "tool": observation.get("name", ""),
                    "status": observation.get("status", ""),
                    "summary": observation.get("summary", ""),
                    "data": observation.get("data", {}),
                    "artifacts": observation.get("artifacts", []),
                    "error": observation.get("error"),
                },
            }
        )

    return {"plan": plan, "tool_trajectory": tool_trajectory, "reflections": reflections}


def _expand_report_args(reports: Iterable[str], successful_reports_file: str = "") -> List[Path]:
    paths = [Path(item) for item in reports]
    if successful_reports_file:
        file_path = Path(successful_reports_file)
        if file_path.exists():
            for line in file_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    paths.append(Path(line.strip()))
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MeteoAgent-DA reports into SFT JSONL samples.")
    parser.add_argument("--reports", nargs="*", default=[])
    parser.add_argument("--successful-reports-file", default="")
    parser.add_argument("--output", default="post_training_sft.jsonl")
    args = parser.parse_args()

    report_paths = _expand_report_args(args.reports, args.successful_reports_file)
    if not report_paths:
        raise SystemExit("--reports or --successful-reports-file is required.")

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for item in report_paths:
            sample = trace_to_sft(item)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote SFT samples to {output}")


if __name__ == "__main__":
    main()
