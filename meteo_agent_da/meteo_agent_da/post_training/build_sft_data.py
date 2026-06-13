from __future__ import annotations

import argparse
import json
from pathlib import Path


def trace_to_sft(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
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
                    {"tool_trajectory": tool_steps, "final_summary": report["summary"]},
                    ensure_ascii=False,
                ),
            },
        ],
        "metadata": {
            "run_id": report["run_id"],
            "status": report["status"],
            "artifacts": report.get("artifacts", []),
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


if __name__ == "__main__":
    main()
