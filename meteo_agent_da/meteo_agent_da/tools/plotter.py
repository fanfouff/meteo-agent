from __future__ import annotations

from itertools import islice

from ..agent.schemas import ProjectConfig, ToolCall, ToolResult, ToolStatus


def run_plotter(call: ToolCall, config: ProjectConfig) -> ToolResult:
    figure_roots = [
        config.project_root / "prediction" / "paper_figures",
        config.project_root / "prediction" / "figures_64_increment",
        config.project_root / "prediction" / "figures_128_increment_final",
    ]
    figures = []
    for root in figure_roots:
        if root.exists():
            for pattern in ("*.png", "*.pdf"):
                figures.extend(str(path) for path in islice(root.glob(pattern), 20))

    suggested_commands = [
        "python prediction/paper_figures.py",
        "python prediction/inf_data/plot_yearlong_level_rmse_acc.py --config prediction/eval_config_64.yaml",
        "python prediction/inf_fig/export_case_maps_attention.py",
    ]
    return ToolResult(
        name=call.name,
        status=ToolStatus.OK,
        summary=f"Registered {len(figures)} existing figure artifact(s) and {len(suggested_commands)} plotting command(s).",
        data={"existing_figures": figures[:40], "suggested_commands": suggested_commands},
        artifacts=figures[:40],
    )
