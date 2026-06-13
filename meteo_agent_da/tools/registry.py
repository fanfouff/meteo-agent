from __future__ import annotations

from ..agent.tool_registry import ToolRegistry, ToolSpec
from .data_indexer import run_data_indexer
from .evaluator import run_evaluator
from .paper_writer import run_paper_writer
from .pasnet_runner import run_pasnet_runner
from .plotter import run_plotter
from .sanity_checker import run_sanity_check


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="sanity_check",
            description="Check PASNet-DA project paths and default resources.",
            handler=run_sanity_check,
        )
    )
    registry.register(
        ToolSpec(
            name="data_indexer",
            description="Inspect FY-3F/ERA5 npz data roots, stats files, and split files.",
            handler=run_data_indexer,
        )
    )
    registry.register(
        ToolSpec(
            name="pasnet_runner",
            description="Build PASNet-DA training/evaluation commands in a controlled form.",
            handler=run_pasnet_runner,
            risky=True,
        )
    )
    registry.register(
        ToolSpec(
            name="evaluator",
            description="Summarize metric, history, and experiment-status artifacts.",
            handler=run_evaluator,
        )
    )
    registry.register(
        ToolSpec(
            name="plotter",
            description="Register existing figures and suggest plotting commands.",
            handler=run_plotter,
        )
    )
    registry.register(
        ToolSpec(
            name="paper_writer",
            description="Generate LaTeX tables, captions, and paper-style wording.",
            handler=run_paper_writer,
        )
    )
    return registry
