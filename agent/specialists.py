"""P2: Specialist 定义与注册表 —— 每个 specialist = prompt + 工具子集。

扩展新 specialist（如 Analyst/Planner/Debugger）只需在此注册表加一条：
supervisor 的 system prompt 与 delegate 工具描述都会自动包含它，
agent/core.py、agent/graph.py、agent/supervisor.py 的装配逻辑零改动。
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.prompts import (
    CODER_SYSTEM_PROMPT,
    REVIEWER_PROMPT,
    SUPERVISOR_PROMPT_TEMPLATE,
    TESTER_PROMPT,
)


@dataclass(frozen=True)
class Specialist:
    name: str                          # delegate 工具里的 specialist 名（小写）
    role: str                          # 一句话职责（supervisor prompt / delegate 描述）
    system_prompt: str
    tool_names: tuple[str, ...]
    max_iterations: int | None = None  # None = 用全局 settings.max_iterations


ALL_TOOL_NAMES = (
    "list_files", "read_file", "write_file", "edit_file",
    "delete_file", "search_code", "run_command",
)
TESTER_TOOL_NAMES = (
    "list_files", "read_file", "write_file", "edit_file",
    "search_code", "run_command",
)
REVIEWER_TOOL_NAMES = ("list_files", "read_file", "search_code")


SPECIALISTS: dict[str, Specialist] = {
    "coder": Specialist(
        name="coder",
        role="编写和修改业务代码，并创建测试",
        system_prompt=CODER_SYSTEM_PROMPT,
        tool_names=ALL_TOOL_NAMES,
    ),
    "tester": Specialist(
        name="tester",
        role="编写/修改 test_*.py 测试并运行 pytest 验证，不改业务代码",
        system_prompt=TESTER_PROMPT,
        tool_names=TESTER_TOOL_NAMES,
    ),
    "reviewer": Specialist(
        name="reviewer",
        role="只读审查代码与测试，输出审查意见，不改任何文件",
        system_prompt=REVIEWER_PROMPT,
        tool_names=REVIEWER_TOOL_NAMES,
    ),
}


def specialist_listing() -> str:
    """渲染 delegate 工具描述 / supervisor prompt 里的 specialist 清单。"""
    return "\n".join(
        f"- {spec.name}: {spec.role}" for spec in SPECIALISTS.values()
    )


def build_supervisor_prompt() -> str:
    """动态生成 supervisor 的 system prompt（嵌入当前 specialist 清单）。"""
    return SUPERVISOR_PROMPT_TEMPLATE.format(specialists=specialist_listing())
