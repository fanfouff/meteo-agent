from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def verify_artifacts(
    report: Dict[str, Any],
    expected_artifacts: Iterable[str] = (),
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    declared = _collect_artifacts(report)
    expected = [str(item) for item in expected_artifacts]
    root = Path(base_dir) if base_dir else Path.cwd()

    existing = []
    missing_declared = []
    for item in declared:
        resolved = _resolve_path(item, root)
        if resolved.exists():
            existing.append(item)
        else:
            missing_declared.append(item)

    missing_expected = [
        item for item in expected if not any(item in artifact for artifact in declared)
    ]
    artifact_recall = 1.0
    if expected:
        artifact_recall = (len(expected) - len(missing_expected)) / len(expected)
    existing_rate = 1.0 if not declared else len(existing) / len(declared)
    hallucination_rate = 0.0 if not declared else len(missing_declared) / len(declared)

    return {
        "pass": artifact_recall == 1.0 and hallucination_rate == 0.0,
        "declared_artifacts": declared,
        "declared_artifact_count": len(declared),
        "existing_artifact_count": len(existing),
        "artifact_recall": artifact_recall,
        "existing_artifact_rate": existing_rate,
        "hallucination_rate": hallucination_rate,
        "missing_expected_artifacts": missing_expected,
        "missing_declared_artifacts": missing_declared,
    }


def _collect_artifacts(report: Dict[str, Any]) -> list[str]:
    seen = set()
    artifacts = []
    for item in report.get("artifacts", []) or []:
        text = str(item)
        if text not in seen:
            seen.add(text)
            artifacts.append(text)
    for result in report.get("tool_results", []) or []:
        for item in result.get("artifacts", []) or []:
            text = str(item)
            if text not in seen:
                seen.add(text)
                artifacts.append(text)
    return artifacts


def _resolve_path(path: str, base_dir: Path) -> Path:
    item = Path(path)
    if item.is_absolute():
        return item
    return base_dir / item
