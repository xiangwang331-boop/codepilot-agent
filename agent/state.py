"""Agent State。

设计要点（对应 OpenHands 的 ConversationState，收敛成一个 TypedDict）：
- `messages` 是**唯一权威真源**，由 LangGraph 的 add_messages reducer 维护，
  自动完成 action(tool_calls) 与 observation(ToolMessage) 的 tool_call_id 配对。
- `tool_calls` / `observations` 只是镜像日志，**只在 tools 节点写一次**，供 CLI/未来 Web 展示。
- 其它是控制字段：状态机 status + 迭代计数（防死循环）+ 结果。
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # —— 会话内容（唯一真源）——
    messages: Annotated[list[BaseMessage], add_messages]

    # —— 控制字段 ——
    current_agent: str                            # 当前 agent 角色名（P0 固定 "Coder"）
    current_task: str                             # 用户原始需求
    iteration_count: int                          # 迭代计数，防死循环
    max_iterations: int
    status: Literal["running", "finished", "error"]
    error: str | None
    result: str | None                            # 最终回答文本

    # —— 镜像日志（只读展示，非真源）——
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    observations: Annotated[list[dict[str, Any]], operator.add]
