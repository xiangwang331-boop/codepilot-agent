"""工具注册表：汇总所有工具，按名查找。

对应 OpenHands 的 register_tool / resolve_tool：启动时解析一次，
之后图执行时按工具名查表。
"""
from __future__ import annotations

from typing import Iterable

from langchain_core.tools import BaseTool

from tools import filesystem, terminal
from workspace.manager import WorkspaceManager


def build_tools(ws: WorkspaceManager) -> list[BaseTool]:
    return filesystem.build_tools(ws) + terminal.build_tools(ws)


def build_tools_map(ws: WorkspaceManager) -> dict[str, BaseTool]:
    return {t.name: t for t in build_tools(ws)}


def build_tools_subset(ws: WorkspaceManager, names: Iterable[str]) -> list[BaseTool]:
    """按名字集合过滤工具，供 P2 specialist 使用；未知名字抛 ValueError（装配期暴露手误）。"""
    full = build_tools_map(ws)
    names = tuple(names)
    unknown = [n for n in names if n not in full]
    if unknown:
        raise ValueError(f"未知工具: {sorted(unknown)}; 可用: {sorted(full)}")
    return [full[n] for n in names]
