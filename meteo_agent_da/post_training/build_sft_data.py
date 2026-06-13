from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def trace_to_sft(report_path: Path, trace_path: Optional[Path] = None) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trace_events = _load_trace(trace_path or report_path.parent / "trace.jsonl")
    tool_steps = []
    for result in report.get("tool_results", []):
        tool_steps.append(
            {
                "tool": result["name"],
                "status": result["status"],
                "summary": result["summary"],
                "artifacts": result.get("artifacts", []),
            }
        )
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
                        "trace_events": _compact_trace(trace_events),
                        "tool_trajectory": tool_steps,
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
            "trace_path": str(trace_path or report_path.parent / "trace.jsonl"),
            "num_trace_events": len(trace_events),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MeteoAgent-DA reports into SFT JSONL samples.")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", default="post_training_sft.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for item in args.reports:
            sample = trace_to_sft(Path(item))
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote SFT samples to {output}")


def _load_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def _compact_trace(events: list[dict]) -> list[dict]:
    compact = []
    for event in events:
        payload = event.get("payload", {})
        item = {
            "step": event.get("step"),
            "kind": event.get("kind"),
            "message": event.get("message"),
        }
        if event.get("kind") == "tool":
            item["tool_call"] = {
                "name": payload.get("name"),
                "arguments": payload.get("arguments", {}),
                "reason": payload.get("reason", ""),
            }
        elif event.get("kind") == "reflect":
            item["observation"] = {
                "name": payload.get("name"),
                "status": payload.get("status"),
                "summary": payload.get("summary"),
                "error": payload.get("error"),
            }
        elif event.get("kind") == "plan":
            item["plan"] = {
                "objective": payload.get("objective"),
                "tool_calls": payload.get("tool_calls", []),
            }
        elif event.get("kind") == "report":
            item["report_path"] = payload.get("report_path")
        compact.append(item)
    return compact


if __name__ == "__main__":
    main()
