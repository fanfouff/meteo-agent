from __future__ import annotations

import re
from pathlib import Path

from .schemas import AgentPlan, AgentTask, ToolCall


class HeuristicPlanner:
    """A deterministic starter planner.

    Later this can be replaced by an LLM planner while keeping the same ToolCall
    contract.
    """

    def plan(self, task: AgentTask) -> AgentPlan:
        request = task.request
        lower = request.lower()
        calls: list[ToolCall] = [
            ToolCall(
                name="sanity_check",
                arguments={},
                reason="Check PASNet project paths and default resources before planning.",
            )
        ]

        if any(token in lower for token in ["data", "dataset", "split", "npz", "数据", "划分", "样本"]):
            calls.append(
                ToolCall(
                    name="data_indexer",
                    arguments={"split_hint": self._extract_split_hint(lower)},
                    reason="Inspect data and split resources for the requested experiment.",
                )
            )

        if any(token in lower for token in ["train", "训练", "pasnet", "swin", "fuxi", "mamba", "实验", "compare", "比较"]):
            calls.append(
                ToolCall(
                    name="pasnet_runner",
                    arguments={
                        "models": self._extract_models(lower),
                        "split_hint": self._extract_split_hint(lower),
                        "epochs": self._extract_int(lower, "epoch", default=200),
                    },
                    reason="Build controlled PASNet-DA experiment commands.",
                )
            )

        if any(token in lower for token in ["rmse", "mae", "metric", "评估", "误差", "指标"]):
            calls.append(
                ToolCall(
                    name="evaluator",
                    arguments={},
                    reason="Summarize available metric artifacts.",
                )
            )

        if any(token in lower for token in ["plot", "figure", "case", "图", "绘图", "可视化"]):
            calls.append(
                ToolCall(
                    name="plotter",
                    arguments={},
                    reason="Prepare figure generation commands or register expected figures.",
                )
            )

        if any(token in lower for token in ["paper", "latex", "table", "论文", "表格", "图注", "caption"]):
            calls.append(
                ToolCall(
                    name="paper_writer",
                    arguments={"style": "ieee"},
                    reason="Generate paper-ready tables and result wording.",
                )
            )

        return AgentPlan(
            objective=request,
            tool_calls=calls,
            notes=[
                "The first scaffold uses a deterministic planner; replace it with an LLM planner after tools stabilize.",
                "Expensive commands are dry-run by default.",
            ],
        )

    @staticmethod
    def _extract_split_hint(text: str) -> str:
        match = re.search(r"(\d+)\s*%|(\d+)pct|split[_ -]?(\d+)", text)
        if not match:
            return "100pct"
        value = next(group for group in match.groups() if group)
        return f"{value}pct"

    @staticmethod
    def _extract_models(text: str) -> list[str]:
        candidates = {
            "pasnet": "pasnet",
            "physics": "physics_unet",
            "swin": "swin_unet",
            "fuxi": "fuxi_da",
            "mamba": "mamba",
            "vanilla": "vanilla_unet",
        }
        found = [model for key, model in candidates.items() if key in text]
        return found or ["pasnet"]

    @staticmethod
    def _extract_int(text: str, key: str, default: int) -> int:
        match = re.search(rf"{key}s?\s*[=: ]\s*(\d+)", text)
        return int(match.group(1)) if match else default
