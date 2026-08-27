"""工具注册表：汇总所有工具，按名查找。

对应 OpenHands 的 register_tool / resolve_tool：启动时解析一次，
之后图执行时按工具名查表。
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from tools import filesystem, terminal
from workspace.manager import WorkspaceManager


def build_tools(ws: WorkspaceManager) -> list[BaseTool]:
    return filesystem.build_tools(ws) + terminal.build_tools(ws)


def build_tools_map(ws: WorkspaceManager) -> dict[str, BaseTool]:
    return {t.name: t for t in build_tools(ws)}
