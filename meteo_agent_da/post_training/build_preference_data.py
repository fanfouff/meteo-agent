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


def reports_to_preference(chosen_path: Path, rejected_path: Path, allow_error_chosen: bool = False) -> Dict[str, Any]:
    chosen = _load(chosen_path)
    rejected = _load(rejected_path)
    if chosen.get("status") != "ok" and not allow_error_chosen:
        raise ValueError(
            f"chosen report must have status=ok; got {chosen.get('status')} from {chosen_path}. "
            "Use --allow-error-chosen only for debugging, not for DPO training data."
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO-style preference data from two MeteoAgent-DA reports.")
    parser.add_argument("--chosen", required=True, nargs="+")
    parser.add_argument("--rejected", required=True, nargs="+")
    parser.add_argument("--output", default="post_training_preferences.jsonl")
    parser.add_argument("--allow-error-chosen", action="store_true", default=False)
    args = parser.parse_args()

    if len(args.chosen) != len(args.rejected):
        raise SystemExit("--chosen and --rejected must have the same length.")

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for chosen, rejected in zip(args.chosen, args.rejected):
            try:
                sample = reports_to_preference(Path(chosen), Path(rejected), allow_error_chosen=args.allow_error_chosen)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote preference samples to {output}")


if __name__ == "__main__":
    main()
