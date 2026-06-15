from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_PASNET_ARGS = {
    "--exp_name",
    "--output_dir",
    "--data_root",
    "--stats_file",
    "--increment_stats",
    "--split_mode",
    "--split_file",
    "--model",
    "--epochs",
    "--batch_size",
    "--lr",
    "--loss",
}

ARGS_WITH_VALUES = {
    "--exp_name",
    "--output_dir",
    "--data_root",
    "--stats_file",
    "--increment_stats",
    "--split_mode",
    "--split_file",
    "--model",
    "--epochs",
    "--batch_size",
    "--lr",
    "--loss",
}

UNSAFE_TOKENS = {"rm", "sudo", "mkfs", "dd", "chmod", "chown", "shutdown", "reboot"}


def verify_commands(report: Dict[str, Any]) -> Dict[str, Any]:
    commands = _extract_pasnet_commands(report)
    pasnet_used = any(
        str(item.get("name", "")) == "pasnet_runner"
        for item in report.get("tool_results", []) or []
    )
    if not pasnet_used and not commands:
        return {
            "pass": True,
            "command_validity": 1.0,
            "command_count": 0,
            "unsafe_command_count": 0,
            "invalid_commands": [],
        }

    if not commands:
        return {
            "pass": False,
            "command_validity": 0.0,
            "command_count": 0,
            "unsafe_command_count": 0,
            "invalid_commands": [{"reason": "pasnet_runner_produced_no_commands"}],
        }

    valid = 0
    unsafe_count = 0
    invalid = []
    for index, command in enumerate(commands):
        args = _command_args(command)
        reasons = []
        missing_args = sorted(REQUIRED_PASNET_ARGS - set(args))
        if missing_args:
            reasons.append({"missing_args": missing_args})
        empty_args = sorted(key for key in ARGS_WITH_VALUES if key in args and not _value_after(args, key))
        if empty_args:
            reasons.append({"empty_args": empty_args})
        split_file = _value_after(args, "--split_file")
        if split_file and Path(split_file).is_absolute() and not Path(split_file).exists():
            reasons.append({"missing_split_file": split_file})
        if _is_unsafe(args):
            unsafe_count += 1
            reasons.append({"unsafe_command": args[:4]})
        if reasons:
            invalid.append({"index": index, "model": command.get("model", ""), "reasons": reasons})
        else:
            valid += 1

    command_validity = valid / len(commands)
    return {
        "pass": command_validity == 1.0 and unsafe_count == 0,
        "command_validity": command_validity,
        "command_count": len(commands),
        "unsafe_command_count": unsafe_count,
        "invalid_commands": invalid,
    }


def _extract_pasnet_commands(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []
    for result in report.get("tool_results", []) or []:
        if str(result.get("name", "")) != "pasnet_runner":
            continue
        data = dict(result.get("data", {}) or {})
        for item in data.get("commands", []) or []:
            if isinstance(item, dict):
                commands.append(item)
    return commands


def _command_args(command: Dict[str, Any]) -> List[str]:
    raw = command.get("command", [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    shell = str(command.get("shell", raw or ""))
    return shlex.split(shell) if shell else []


def _value_after(args: List[str], key: str) -> str:
    try:
        index = args.index(key)
    except ValueError:
        return ""
    if index + 1 >= len(args):
        return ""
    value = args[index + 1]
    if value.startswith("--"):
        return ""
    return value


def _is_unsafe(args: List[str]) -> bool:
    tokens = {Path(item).name for item in args[:3]}
    if tokens.intersection(UNSAFE_TOKENS):
        return True
    joined = " ".join(args).lower()
    return "curl " in joined and " bash" in joined
