from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable

from .schemas import ProjectConfig, ToolCall, ToolResult, ToolStatus


ToolHandler = Callable[[ToolCall, ProjectConfig], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: ToolHandler
    risky: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def names(self) -> Iterable[str]:
        return self._tools.keys()

    def specs(self) -> Iterable[ToolSpec]:
        return self._tools.values()

    def run(self, call: ToolCall, config: ProjectConfig) -> ToolResult:
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(
                name=call.name,
                status=ToolStatus.ERROR,
                summary=f"Unknown tool: {call.name}",
                error="unknown_tool",
            )
        if spec.risky and config.dry_run is False and call.arguments.get("allow_risky") is not True and not config.allow_risky_tools:
            return ToolResult(
                name=call.name,
                status=ToolStatus.ERROR,
                summary=f"Risky tool {call.name} requires allow_risky=true.",
                error="approval_required",
            )
        return spec.handler(call, config)
