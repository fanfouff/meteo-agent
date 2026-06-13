from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .schemas import ToolResult


@dataclass
class WorkingMemory:
    """Small, explicit memory for a single research run."""

    task_summary: str = ""
    observations: List[str] = field(default_factory=list)
    tool_summaries: Dict[str, str] = field(default_factory=dict)

    def add_observation(self, text: str) -> None:
        if text and text not in self.observations:
            self.observations.append(text)

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_summaries[result.name] = result.summary
        self.add_observation(result.summary)

    def snapshot(self) -> Dict[str, object]:
        return {
            "task_summary": self.task_summary,
            "observations": list(self.observations[-10:]),
            "tool_summaries": dict(self.tool_summaries),
        }
