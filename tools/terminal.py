"""终端工具：run_command —— 在 workspace 根目录内执行 shell 命令。

P5 起执行宿主抽到 CommandRunner（tools/command_runner.py）：默认本机 subprocess
（LocalCommandRunner），SANDBOX_MODE=docker 时注入 DockerCommandRunner 把命令丢进
容器（host workspace 经 bind mount 成 /workspace）。这里仍是唯一暴露 run_command
给 agent 的工具封装；文件工具仍走 WorkspaceManager，本文件不碰文件。
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.command_runner import DockerCommandRunner, LocalCommandRunner
from workspace.manager import WorkspaceManager


class RunCommandArgs(BaseModel):
    command: str = Field(..., description="要执行的 shell 命令，如 'pytest' 或 'python main.py'")
    timeout: int = Field(default=60, description="超时秒数")


def build_tools(ws: WorkspaceManager, *, runner=None) -> list[StructuredTool]:
    """构造 run_command 工具。

    runner: 执行宿主。None → LocalCommandRunner(ws.root)（本机 subprocess，默认行为，
    与 P0–P4 完全一致）。传 DockerCommandRunner 即进入沙箱模式（运行隔离）。
    """
    runner = runner or LocalCommandRunner(ws.root)

    def fn(command: str, timeout: int = 60) -> str:
        return runner.run(command, timeout)

    description = (
        "在 workspace 根目录内执行 shell 命令（pytest / python 等），"
        "返回退出码、stdout、stderr。用于运行测试和验证代码。"
    )
    if isinstance(runner, DockerCommandRunner):
        # docker 模式：命令跑在容器内 /workspace（= host workspace bind mount），
        # 引导模型把输出里的 /workspace/xxx 当相对路径 xxx，文件工具一律用相对路径。
        description += (
            "当前命令在 Docker 容器 /workspace 内执行，与 workspace 根目录一一对应；"
            "命令输出中的 /workspace/xxx 等价于相对路径 xxx，文件工具请一律用相对路径。"
        )
    return [
        StructuredTool.from_function(
            name="run_command",
            description=description,
            args_schema=RunCommandArgs,
            func=fn,
        ),
    ]
