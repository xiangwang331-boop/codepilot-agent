"""P7: API 的请求/响应模型（Pydantic）。

只做**形状与校验**，不含业务逻辑——状态机在 `runtime/session.py`，这里连
「能不能 begin」都不知道（那是 `Session.begin()` 返回 False 的事，由 routes 翻成 409）。

`SessionInfo` 刻意与 `Session.snapshot()` 的键逐一对齐（外加 WS 的 `status` 信封）：
三处（REST / WS 状态信封 / snapshot）形状一致，前端一份解析即可。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    thread_id: str | None = Field(
        default=None,
        description="自定义会话 ID；留空自动生成 UUID。会话的工作目录是 "
        "`<WORKSPACE_ROOT>/<会话ID>`，容器 bind mount 的也是它。",
    )


class TaskRequest(BaseModel):
    task: str = Field(min_length=1, description="开发需求（自然语言）")


class ApprovalRequest(BaseModel):
    approved: bool = Field(
        description="true = 批准挂起的委派，false = 拒绝。服务端会精确映射成 "
        "'yes'/'no' 再喂给 LangGraph（见 runtime/driver.normalize_answer）"
    )


class SessionInfo(BaseModel):
    """会话状态快照。"""

    thread_id: str
    status: str = Field(description="idle | running | awaiting_approval | closed")
    approval: list[dict[str, Any]] = Field(
        default_factory=list, description="待批准的问题（status=awaiting_approval 时非空）"
    )
    result: dict[str, Any] | None = Field(default=None, description="最近一次任务的最终 state")
    error: str | None = None
    event_count: int = 0


class SessionList(BaseModel):
    sessions: list[SessionInfo]


class EventEnvelope(BaseModel):
    """一条事件信封（与 WS 推的 `kind=event` 逐字段相同）。"""

    seq: int = Field(description="会话内序号（从 0 起，单调递增；WS 断线重连传 `?since=`）")
    event: dict[str, Any]


class EventList(BaseModel):
    thread_id: str
    events: list[EventEnvelope]


class ErrorDetail(BaseModel):
    """409 / 400 的 `detail`（FastAPI 会包成 `{"detail": {...}}`）。"""

    status: str = Field(description="会话当前状态，前端据此决定显示什么按钮")
    message: str


__all__ = [
    "ApprovalRequest",
    "CreateSessionRequest",
    "ErrorDetail",
    "EventEnvelope",
    "EventList",
    "SessionInfo",
    "SessionList",
    "TaskRequest",
]
