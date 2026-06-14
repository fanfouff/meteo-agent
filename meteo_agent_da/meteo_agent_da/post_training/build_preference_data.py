from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assistant_content(report: Dict[str, Any]) -> str:
    trajectory = [
        {
            "tool": result.get("name"),
            "status": result.get("status"),
            "summary": result.get("summary"),
            "artifacts": result.get("artifacts", []),
        }
        for result in report.get("tool_results", [])
    ]
    return json.dumps(
        {
            "tool_trajectory": trajectory,
            "final_summary": report.get("summary", ""),
            "artifacts": report.get("artifacts", []),
        },
        ensure_ascii=False,
    )


def reports_to_preference(chosen_path: Path, rejected_path: Path) -> Dict[str, Any]:
    chosen = _load(chosen_path)
    rejected = _load(rejected_path)
    prompt = chosen.get("request") or rejected.get("request", "")
    return {
        "prompt": [
            {
                "role": "system",
                "content": "You are MeteoAgent-DA. Prefer executable, tool-grounded satellite data-assimilation workflows over unsupported text-only answers.",
            },
            {"role": "user", "content": prompt},
        ],
        "chosen": [{"role": "assistant", "content": _assistant_content(chosen)}],
        "rejected": [{"role": "assistant", "content": _assistant_content(rejected)}],
        "metadata": {
            "chosen_report": str(chosen_path),
            "rejected_report": str(rejected_path),
            "chosen_status": chosen.get("status"),
            "rejected_status": rejected.get("status"),
        },
    }


def pairs_from_scores(scores_path: Path, chosen_method: str, rejected_method: str, require_chosen_pass: bool = True) -> list[tuple[Path, Path]]:
    payload = _load(scores_path)
    rows = list(payload.get("scores", []))
    chosen_by_task = {}
    rejected_by_task = {}
    for row in rows:
        key = (row.get("task_id"), row.get("attempt", 1))
        if row.get("method") == chosen_method:
            if require_chosen_pass and float(row.get("pass_rate", 0.0)) < 1.0:
                continue
            chosen_by_task[key] = Path(row["report"])
        if row.get("method") == rejected_method:
            rejected_by_task[key] = Path(row["report"])
    pairs = []
    for key, chosen_path in chosen_by_task.items():
        rejected_path = rejected_by_task.get(key)
        if rejected_path:
            pairs.append((chosen_path, rejected_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO-style preference data from two MeteoAgent-DA reports.")
    parser.add_argument("--chosen", nargs="*", default=[])
    parser.add_argument("--rejected", nargs="*", default=[])
    parser.add_argument("--scores-json", default="")
    parser.add_argument("--chosen-method", default="heuristic_tool")
    parser.add_argument("--rejected-method", default="text_only")
    parser.add_argument("--allow-failed-chosen", action="store_true", default=False)
    parser.add_argument("--output", default="post_training_preferences.jsonl")
    args = parser.parse_args()

    if args.scores_json:
        pairs = pairs_from_scores(
            Path(args.scores_json),
            chosen_method=args.chosen_method,
            rejected_method=args.rejected_method,
            require_chosen_pass=not args.allow_failed_chosen,
        )
    else:
        if len(args.chosen) != len(args.rejected):
            raise SystemExit("--chosen and --rejected must have the same length.")
        pairs = [(Path(chosen), Path(rejected)) for chosen, rejected in zip(args.chosen, args.rejected)]

    if not pairs:
        raise SystemExit("No preference pairs were produced.")

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for chosen, rejected in pairs:
            sample = reports_to_preference(chosen, rejected)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote preference samples to {output}")


if __name__ == "__main__":
    main()
