from __future__ import annotations

import csv
import json
from itertools import islice
from pathlib import Path

from ..agent.schemas import ProjectConfig, ToolCall, ToolResult, ToolStatus


def run_evaluator(call: ToolCall, config: ProjectConfig) -> ToolResult:
    roots = [
        config.project_root / "prediction",
        config.project_root / "train_ddp" / "outputs",
        config.default_output_dir,
    ]
    metric_files: list[Path] = []
    for root in roots:
        if root.exists():
            metric_files.extend(islice(root.glob("**/metrics.json"), 20))
            metric_files.extend(islice(root.glob("**/history.csv"), 20))
            metric_files.extend(islice(root.glob("**/experiment_status.csv"), 20))

    summaries = []
    for path in metric_files[:20]:
        summaries.append(_summarize_metric_file(path))

    summary = f"Found {len(metric_files)} metric/status artifact(s); summarized {len(summaries)}."
    return ToolResult(
        name=call.name,
        status=ToolStatus.OK,
        summary=summary,
        data={"metric_files": [str(path) for path in metric_files[:20]], "summaries": summaries},
        artifacts=[str(path) for path in metric_files[:20]],
    )


def _summarize_metric_file(path: Path) -> dict:
    if path.suffix == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            keys = ["rmse", "mae", "corr", "acc", "global_rmse", "strat_rmse", "trop_rmse"]
            return {"path": str(path), "values": {key: raw.get(key) for key in keys if key in raw}}
        except Exception as exc:  # pragma: no cover - defensive summary
            return {"path": str(path), "error": str(exc)}

    try:
        with path.open("r", encoding="utf-8") as f:
            rows = list(islice(csv.DictReader(f), 5))
        return {"path": str(path), "rows_preview": rows}
    except Exception as exc:  # pragma: no cover
        return {"path": str(path), "error": str(exc)}
