from __future__ import annotations

from pathlib import Path

from ..agent.schemas import ProjectConfig, ToolCall, ToolResult, ToolStatus
from .data_indexer import _resolve_split_file


def run_pasnet_runner(call: ToolCall, config: ProjectConfig) -> ToolResult:
    models = call.arguments.get("models") or ["pasnet"]
    split_hint = str(call.arguments.get("split_hint") or "100pct")
    epochs = int(call.arguments.get("epochs") or 200)
    split_file = _resolve_split_file(config, split_hint)

    commands = []
    for model in models:
        model_name = _normalize_model_name(str(model))
        exp_name = f"{model_name}_split_{split_hint}"
        cmd = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(config.resolve_project_path(config.train_script)),
            "--exp_name",
            exp_name,
            "--output_dir",
            str(config.default_output_dir),
            "--data_root",
            str(config.default_data_root),
            "--stats_file",
            str(config.default_stats_file),
            "--increment_stats",
            str(config.default_increment_stats),
            "--split_mode",
            "file",
            "--split_file",
            str(split_file),
            "--save_split",
            "false",
            "--model",
            model_name,
            "--fusion_mode",
            "gated",
            "--use_aux",
            "false",
            "--mask_aware",
            "true",
            "--use_spectral_stem",
            "true",
            "--deep_supervision",
            "true",
            "--epochs",
            str(epochs),
            "--batch_size",
            "8",
            "--grad_accum_steps",
            "2",
            "--lr",
            "0.0001",
            "--weight_decay",
            "1e-5",
            "--scheduler",
            "cosine",
            "--loss",
            "combined",
            "--use_increment",
            "--amp",
            "false",
            "--tensorboard",
            "true",
            "--find_unused_parameters",
            "false",
        ]
        commands.append({"model": model_name, "exp_name": exp_name, "command": cmd, "shell": " ".join(cmd)})

    summary = f"Built {len(commands)} PASNet-DA command(s) for split {split_hint}; dry_run={config.dry_run}."
    return ToolResult(
        name=call.name,
        status=ToolStatus.OK,
        summary=summary,
        data={
            "commands": commands,
            "split_file": str(split_file),
            "execute_note": "Commands are not executed by the scaffold. Review GPU and paths first.",
        },
    )


def _normalize_model_name(model: str) -> str:
    aliases = {
        "pasnet": "pasnet",
        "physics": "physics_unet",
        "physics_unet": "physics_unet",
        "swin": "swin_unet",
        "swin_unet": "swin_unet",
        "fuxi": "fuxi_da",
        "fuxi_da": "fuxi_da",
        "mamba": "mamba",
        "vanilla": "vanilla_unet",
        "vanilla_unet": "vanilla_unet",
    }
    return aliases.get(model.lower(), model)
