from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Optional

from ..agent.schemas import ProjectConfig, ToolCall, ToolResult, ToolStatus


def run_data_indexer(call: ToolCall, config: ProjectConfig) -> ToolResult:
    data_root = Path(call.arguments.get("data_root") or config.default_data_root)
    split_hint = str(call.arguments.get("split_hint") or "100pct")
    split_file = _resolve_split_file(config, split_hint)

    data = {
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "split_hint": split_hint,
        "split_file": str(split_file) if split_file else "",
        "split_file_exists": bool(split_file and split_file.exists()),
        "stats_file": str(config.default_stats_file),
        "stats_file_exists": config.default_stats_file.exists(),
        "increment_stats": str(config.default_increment_stats),
        "increment_stats_exists": config.default_increment_stats.exists(),
    }

    if data_root.exists():
        sample_files = list(islice(data_root.glob("**/*.npz"), 20))
        data["sample_npz_count_first_20"] = len(sample_files)
        data["sample_npz_files"] = [str(path) for path in sample_files[:5]]

    if split_file and split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        data["split_counts"] = {key: len(raw.get(key, [])) for key in ("train", "val", "test")}

    summary = (
        f"Indexed data_root={data_root}; split={split_hint}; "
        f"split_file_exists={data['split_file_exists']}."
    )
    status = ToolStatus.OK if data["data_root_exists"] and data["split_file_exists"] else ToolStatus.ERROR
    error = None if status == ToolStatus.OK else "missing_data_root_or_split_file"
    return ToolResult(name=call.name, status=status, summary=summary, data=data, error=error)


def _resolve_split_file(config: ProjectConfig, split_hint: str) -> Optional[Path]:
    split_dir = config.default_split_dir
    candidates = [
        split_dir / f"split_{split_hint}.json",
        split_dir / f"{split_hint}.json",
        config.project_root / "train_ddp" / "splits" / f"fixed_split_{split_hint.replace('pct', '')}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
