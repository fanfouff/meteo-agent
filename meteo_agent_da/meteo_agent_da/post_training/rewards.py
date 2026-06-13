from __future__ import annotations

from typing import Iterable, Mapping


def executable_workflow_reward(
    tool_results: Iterable[Mapping[str, object]],
    required_artifacts: Iterable[str] = (),
    scientific_checks: Mapping[str, bool] | None = None,
) -> float:
    results = list(tool_results)
    if not results:
        return 0.0

    success_rate = sum(1 for item in results if item.get("status") == "ok") / len(results)
    artifacts = {artifact for item in results for artifact in item.get("artifacts", [])}
    required = set(required_artifacts)
    artifact_score = 1.0 if not required else len(required & artifacts) / len(required)
    checks = scientific_checks or {}
    scientific_score = 1.0 if not checks else sum(1 for ok in checks.values() if ok) / len(checks)

    return 0.45 * success_rate + 0.25 * artifact_score + 0.30 * scientific_score
