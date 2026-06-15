from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ..verifiers import verify_report_quality


PREFERENCE_TYPES = {
    "tool_grounded_vs_text_only",
    "verified_report_vs_unverified_report",
    "successful_recovery_vs_failed_recovery",
    "concise_reproducible_vs_verbose_hallucinated",
}


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
            "next_steps": report.get("next_steps", []),
        },
        ensure_ascii=False,
    )


def reports_to_preference(chosen_path: Path, rejected_path: Path, preference_type: str = "", rationale: str = "") -> Dict[str, Any]:
    chosen = _load(chosen_path)
    rejected = _load(rejected_path)
    prompt = chosen.get("request") or rejected.get("request", "")
    case_type = preference_type or infer_preference_type(chosen, rejected)
    if case_type not in PREFERENCE_TYPES:
        raise ValueError(f"Unknown preference_type: {case_type}")
    chosen_quality = verify_report_quality(chosen, base_dir=str(chosen_path.parent))
    rejected_quality = verify_report_quality(rejected, base_dir=str(rejected_path.parent))
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
            "preference_type": case_type,
            "rationale": rationale or _preference_rationale(case_type),
            "chosen_report": str(chosen_path),
            "rejected_report": str(rejected_path),
            "chosen_status": chosen.get("status"),
            "rejected_status": rejected.get("status"),
            "chosen_quality": _compact_quality(chosen_quality),
            "rejected_quality": _compact_quality(rejected_quality),
        },
    }


def infer_preference_type(chosen: Dict[str, Any], rejected: Dict[str, Any]) -> str:
    chosen_tools = list(chosen.get("tool_results", []) or [])
    rejected_tools = list(rejected.get("tool_results", []) or [])
    if chosen_tools and not rejected_tools:
        return "tool_grounded_vs_text_only"
    if chosen.get("status") == "ok" and rejected.get("status") != "ok":
        return "successful_recovery_vs_failed_recovery"

    chosen_artifacts = list(chosen.get("artifacts", []) or [])
    rejected_artifacts = list(rejected.get("artifacts", []) or [])
    if chosen_artifacts and not rejected_artifacts:
        return "verified_report_vs_unverified_report"

    rejected_text = json.dumps(rejected, ensure_ascii=False).lower()
    if "hallucinat" in rejected_text or "missing" in rejected_text or "not found" in rejected_text:
        return "concise_reproducible_vs_verbose_hallucinated"
    return "verified_report_vs_unverified_report"


def _preference_rationale(case_type: str) -> str:
    rationales = {
        "tool_grounded_vs_text_only": "Chosen answer uses executable tool evidence; rejected answer is unsupported text-only reasoning.",
        "verified_report_vs_unverified_report": "Chosen answer has verifier-clean artifacts/commands/metrics; rejected answer lacks verification.",
        "successful_recovery_vs_failed_recovery": "Chosen trajectory recovers from an error or avoids a failed path; rejected trajectory remains failed.",
        "concise_reproducible_vs_verbose_hallucinated": "Chosen report is concise and reproducible; rejected report is verbose, ungrounded, or hallucinated.",
    }
    return rationales[case_type]


def _compact_quality(quality: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verifier_pass": quality.get("verifier_pass", False),
        "failed_checks": quality.get("failed_checks", []),
        "used_tools": quality.get("used_tools", []),
        "artifact_hallucination_rate": quality.get("artifact", {}).get("hallucination_rate", 0.0),
        "command_validity": quality.get("command", {}).get("command_validity", 1.0),
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
    parser.add_argument("--preference-type", choices=sorted(PREFERENCE_TYPES), default="")
    parser.add_argument("--rationale", default="")
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
            sample = reports_to_preference(chosen, rejected, preference_type=args.preference_type, rationale=args.rationale)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote preference samples to {output}")


if __name__ == "__main__":
    main()
