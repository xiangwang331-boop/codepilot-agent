"""终端工具：run_command —— 在 workspace 根目录内执行 shell 命令。

这是唯一允许碰 subprocess 的地方，且 cwd 被锁死在 workspace 根目录。
把当前解释器目录（venv/Scripts）加进 PATH，让 agent 能直接跑 `python` / `pytest`。
返回退出码 + stdout + stderr，供 LLM 判断测试是否通过。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from workspace.manager import WorkspaceManager


class RunCommandArgs(BaseModel):
    command: str = Field(..., description="要执行的 shell 命令，如 'pytest' 或 'python main.py'")
    timeout: int = Field(default=60, description="超时秒数")


def _env_with_runtime_python() -> dict:
    """把当前 Python 解释器目录放到 PATH 最前，保证 python/pytest 可用。"""
    env = os.environ.copy()
    scripts_dir = str(Path(sys.executable).parent)
    env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
    return env


def build_tools(ws: WorkspaceManager) -> list[StructuredTool]:
    def fn(command: str, timeout: int = 60) -> str:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(ws.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                env=_env_with_runtime_python(),
            )
            out: list[str] = []
            if proc.stdout:
                out.append(proc.stdout.rstrip())
            if proc.stderr:
                out.append("[stderr]\n" + proc.stderr.rstrip())
            body = "\n".join(out) if out else "(无输出)"
            return f"exit_code={proc.returncode}\n{body}"
        except subprocess.TimeoutExpired:
            return f"ERROR: 命令超时（>{timeout}s）: {command}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"

    return [
        StructuredTool.from_function(
            name="run_command",
            description="在 workspace 根目录内执行 shell 命令（pytest / python 等），返回退出码、stdout、stderr。用于运行测试和验证代码。",
            args_schema=RunCommandArgs,
            func=fn,
        ),
    ]
