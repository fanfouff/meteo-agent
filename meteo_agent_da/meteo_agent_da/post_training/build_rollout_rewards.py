from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .rewards import pasbench_reward_breakdown


def build_rollout_reward_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    report_path = Path(row["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "prompt": report.get("request", ""),
        "rollout": {
            "method": row.get("method", ""),
            "status": report.get("status", ""),
            "summary": report.get("summary", ""),
            "tool_results": report.get("tool_results", []),
            "artifacts": report.get("artifacts", []),
        },
        "reward": pasbench_reward_breakdown(row),
        "metadata": {
            "task_id": row.get("task_id", ""),
            "category": row.get("category", ""),
            "domain_profile": row.get("domain_profile", ""),
            "attempt": row.get("attempt", 1),
            "report_path": str(report_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GRPO-ready rollout/reward JSONL from PASBench scores.")
    parser.add_argument("--scores-json", required=True)
    parser.add_argument("--output", default="rollout_reward.jsonl")
    parser.add_argument("--require-report-exists", action="store_true", default=False)
    args = parser.parse_args()

    payload = json.loads(Path(args.scores_json).read_text(encoding="utf-8"))
    rows = list(payload.get("scores", []) or [])
    output = Path(args.output)
    written = 0
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            report_path = Path(row.get("report", ""))
            if not report_path.exists():
                if args.require_report_exists:
                    raise SystemExit(f"Report missing: {report_path}")
                continue
            sample = build_rollout_reward_sample(row)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"written": written, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
