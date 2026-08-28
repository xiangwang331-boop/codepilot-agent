"""事件系统。

从第一版就记录事件，未来 Web UI 直接消费同一套事件流。
P0 只做线性日志（刻意舍弃 OpenHands 的事件树 parent_id 分支）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    AGENT_STARTED = "AgentStarted"
    AGENT_STEP = "AgentStep"
    TOOL_CALL_STARTED = "ToolCallStarted"
    TOOL_CALL_COMPLETED = "ToolCallCompleted"
    TOOL_CALL_FAILED = "ToolCallFailed"
    AGENT_COMPLETED = "AgentCompleted"
    AGENT_FAILED = "AgentFailed"
    CONDENSE = "Condense"
    TOKEN_USAGE = "TokenUsage"


@dataclass
class Event:
    type: EventType
    agent: str
    message: str
    detail: dict[str, Any] | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventEmitter:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._listeners: list[Callable[[Event], None]] = []

    def add_listener(self, fn: Callable[[Event], None]) -> None:
        self._listeners.append(fn)

    def emit(
        self, type: EventType, agent: str, message: str, detail: dict | None = None
    ) -> Event:
        e = Event(type=type, agent=agent, message=message, detail=detail)
        self.events.append(e)
        for fn in self._listeners:
            fn(e)
        return e

    def clear(self) -> None:
        self.events.clear()


# 模块级单例：图节点与 main 共用
emitter = EventEmitter()


def emit(type: EventType, agent: str, message: str, detail: dict | None = None) -> Event:
    return emitter.emit(type, agent, message, detail)


def format_event(e: Event) -> str:
    """把事件渲染成 CLI 一行，风格对齐 OpenHands 的 agent/tool 前缀。"""
    prefix = f"[{e.agent}]"
    if e.type is EventType.AGENT_STARTED:
        return f"{prefix} 开始任务"
    if e.type is EventType.AGENT_STEP:
        return f"{prefix} {e.message}"
    if e.type is EventType.TOOL_CALL_STARTED:
        return f"{prefix} 调用 tool: {e.message}"
    if e.type is EventType.TOOL_CALL_COMPLETED:
        return f"    [Tool] {e.message}"
    if e.type is EventType.TOOL_CALL_FAILED:
        return f"    [Tool] ✗ {e.message}"
    if e.type is EventType.AGENT_COMPLETED:
        return f"{prefix} 完成"
    if e.type is EventType.AGENT_FAILED:
        return f"{prefix} 失败：{e.message}"
    if e.type is EventType.CONDENSE:
        out = f"    [Condense] {e.message}"
        summary = (e.detail or {}).get("summary")
        if summary:
            out += "\n" + "\n".join("    | " + line for line in summary.splitlines())
        return out
    if e.type is EventType.TOKEN_USAGE:
        d = e.detail or {}
        return (
            f"{prefix} 消耗 {d.get('total_tokens', '?')} tokens"
            f"（输入 {d.get('prompt_tokens', '?')} → 输出 {d.get('completion_tokens', '?')}）"
        )
    return f"{prefix} {e.message}"
