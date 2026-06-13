from __future__ import annotations

from ..agent.schemas import ProjectConfig, ToolCall, ToolResult, ToolStatus


def run_sanity_check(call: ToolCall, config: ProjectConfig) -> ToolResult:
    project_root = config.project_root
    train_script = config.resolve_project_path(config.train_script)
    checks = {
        "project_root_exists": project_root.exists(),
        "train_script_exists": train_script.exists(),
        "default_data_root_exists": config.default_data_root.exists(),
        "default_stats_file_exists": config.default_stats_file.exists(),
        "default_increment_stats_exists": config.default_increment_stats.exists(),
        "default_split_dir_exists": config.default_split_dir.exists(),
        "dry_run": config.dry_run,
    }
    missing = [key for key, value in checks.items() if key.endswith("_exists") and not value]
    status = ToolStatus.OK if not missing else ToolStatus.ERROR
    summary = "PASNet project sanity check passed." if not missing else "Missing resources: " + ", ".join(missing)
    return ToolResult(
        name=call.name,
        status=status,
        summary=summary,
        data={
            "project_root": str(project_root),
            "train_script": str(train_script),
            "checks": checks,
        },
    )
