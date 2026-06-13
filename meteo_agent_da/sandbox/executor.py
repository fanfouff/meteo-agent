from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass
class CommandResult:
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandExecutor:
    """Small command runner with cwd and timeout control."""

    def __init__(self, cwd: Path, timeout_seconds: int = 120) -> None:
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds

    def run(self, command: list[str], env: Optional[Mapping[str, str]] = None) -> CommandResult:
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.cwd),
                env=dict(env) if env else None,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return CommandResult(
                command=command,
                cwd=str(self.cwd),
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                cwd=str(self.cwd),
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )
