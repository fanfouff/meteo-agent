from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..verifiers import verify_report_quality


def evaluate_trace_filter(
    report_path: Path,
    expected_artifacts: Iterable[str] = (),
    required_tools: Iterable[str] = (),
    require_artifacts: bool = True,
    max_events: int = 80,
    max_trace_chars: int = 120_000,
    dry_run_expected: Optional[bool] = None,
) -> Dict[str, Any]:
    reasons: List[str] = []
    report, report_error = _load_json(report_path)
    if report_error:
        return {
            "accepted": False,
            "report_path": str(report_path),
            "trace_path": "",
            "reasons": [report_error],
            "quality": {},
        }

    trace_path = report_path.with_name("trace.jsonl")
    trace_events, trace_error, trace_chars = _load_trace(trace_path)
    if trace_error:
        reasons.append(trace_error)
    if len(trace_events) > max_events:
        reasons.append("trace_too_long")
    if trace_chars > max_trace_chars:
        reasons.append("trace_chars_too_long")
    if _duplicate_tool_calls(trace_events):
        reasons.append("duplicate_tool_call")

    tool_results = list(report.get("tool_results", []) or [])
    failed_tools = [
        str(item.get("name", ""))
        for item in tool_results
        if str(item.get("status", "")) != "ok"
    ]
    if report.get("status") != "ok":
        reasons.append("report_status_not_ok")
    if not tool_results:
        reasons.append("no_tool_evidence")
    if failed_tools:
        reasons.append("tool_call_failed")

    declared_artifacts = _collect_artifacts(report)
    if require_artifacts and not declared_artifacts:
        reasons.append("no_artifacts")

    quality = verify_report_quality(
        report,
        expected_artifacts=expected_artifacts,
        required_tools=required_tools,
        dry_run_expected=dry_run_expected,
        base_dir=str(report_path.parent),
    )
    for check in quality.get("failed_checks", []):
        reasons.append(f"quality:{check}")

    unique_reasons = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique_reasons.append(reason)

    return {
        "accepted": not unique_reasons,
        "report_path": str(report_path),
        "trace_path": str(trace_path),
        "run_id": report.get("run_id", ""),
        "reasons": unique_reasons,
        "trace_event_count": len(trace_events),
        "trace_char_count": trace_chars,
        "artifact_count": len(declared_artifacts),
        "quality": quality,
    }


def filter_reports(
    report_paths: Iterable[Path],
    expected_artifacts: Iterable[str] = (),
    required_tools: Iterable[str] = (),
    require_artifacts: bool = True,
    max_events: int = 80,
    max_trace_chars: int = 120_000,
    dry_run_expected: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    return [
        evaluate_trace_filter(
            path,
            expected_artifacts=expected_artifacts,
            required_tools=required_tools,
            require_artifacts=require_artifacts,
            max_events=max_events,
            max_trace_chars=max_trace_chars,
            dry_run_expected=dry_run_expected,
        )
        for path in report_paths
    ]


def _load_json(path: Path) -> Tuple[Dict[str, Any], str]:
    if not path.exists():
        return {}, "report_missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except json.JSONDecodeError as exc:
        return {}, f"report_malformed:{exc.lineno}"


def _load_trace(path: Path) -> Tuple[List[Dict[str, Any]], str, int]:
    if not path.exists():
        return [], "trace_missing", 0
    text = path.read_text(encoding="utf-8")
    events: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            return events, f"trace_malformed:{line_number}", len(text)
    return events, "", len(text)


def _duplicate_tool_calls(trace_events: List[Dict[str, Any]]) -> bool:
    seen = set()
    for event in trace_events:
        if event.get("kind") != "tool":
            continue
        payload = dict(event.get("payload", {}) or {})
        key = json.dumps(
            {"name": payload.get("name", ""), "arguments": payload.get("arguments", {})},
            sort_keys=True,
            ensure_ascii=False,
        )
        if key in seen:
            return True
        seen.add(key)
    return False


def _collect_artifacts(report: Dict[str, Any]) -> List[str]:
    artifacts = [str(item) for item in report.get("artifacts", []) or []]
    for result in report.get("tool_results", []) or []:
        artifacts.extend(str(item) for item in result.get("artifacts", []) or [])
    return artifacts


def _expand_report_args(reports: Iterable[str], report_files: Iterable[str]) -> List[Path]:
    paths = [Path(item) for item in reports]
    for file_name in report_files:
        file_path = Path(file_name)
        if not file_path.exists():
            continue
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
    parser = argparse.ArgumentParser(description="Filter MeteoAgent-DA traces before SFT/DPO/GRPO data export.")
    parser.add_argument("--reports", nargs="*", default=[])
    parser.add_argument("--reports-file", action="append", default=[])
    parser.add_argument("--successful-reports-file", default="")
    parser.add_argument("--required-tool", action="append", default=[])
    parser.add_argument("--expected-artifact", action="append", default=[])
    parser.add_argument("--allow-no-artifacts", action="store_true", default=False)
    parser.add_argument("--dry-run-expected", action="store_true", default=False)
    parser.add_argument("--max-events", type=int, default=80)
    parser.add_argument("--max-trace-chars", type=int, default=120_000)
    parser.add_argument("--output", default="filtered_reports.txt")
    parser.add_argument("--decisions-jsonl", default="")
    args = parser.parse_args()

    report_files = list(args.reports_file)
    if args.successful_reports_file:
        report_files.append(args.successful_reports_file)
    report_paths = _expand_report_args(args.reports, report_files)
    if not report_paths:
        raise SystemExit("--reports, --reports-file, or --successful-reports-file is required.")

    decisions = filter_reports(
        report_paths,
        expected_artifacts=args.expected_artifact,
        required_tools=args.required_tool,
        require_artifacts=not args.allow_no_artifacts,
        max_events=args.max_events,
        max_trace_chars=args.max_trace_chars,
        dry_run_expected=True if args.dry_run_expected else None,
    )

    output = Path(args.output)
    accepted = [item["report_path"] for item in decisions if item["accepted"]]
    output.write_text("\n".join(accepted) + ("\n" if accepted else ""), encoding="utf-8")

    decisions_path = Path(args.decisions_jsonl) if args.decisions_jsonl else output.with_suffix(".decisions.jsonl")
    with decisions_path.open("w", encoding="utf-8") as f:
        for item in decisions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({"accepted": len(accepted), "rejected": len(decisions) - len(accepted), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
