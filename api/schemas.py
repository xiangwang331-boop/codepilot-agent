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
    """会话状态快照。

    ⚠️ **字段集是前端契约**：`tests/test_web_ui_contract.py` 对着它做**精确相等**断言
    （`set(body) == SESSION_FIELDS`，REST 单会话与 WS status 信封两处）。加字段 =
    同时改那个常量与 `web/src/api/types.ts`，别只改这里。
    """

    thread_id: str
    status: str = Field(
        description="idle | running | awaiting_approval | interrupted | closed"
        "（interrupted 是 P9：服务重启时正在跑、恢复后只能看历史，见 runtime/catalog.py）"
    )
    approval: list[dict[str, Any]] = Field(
        default_factory=list, description="待批准的问题（status=awaiting_approval 时非空）"
    )
    result: dict[str, Any] | None = Field(default=None, description="最近一次任务的最终 state")
    error: str | None = None
    event_count: int = 0


class SessionList(BaseModel):
    """会话目录（P9 起含**重启后从持久化恢复**的历史会话）。

    `history_available` 是**列表级**能力位，不是每个会话的属性：它说的是
    「这个进程读不读得到历史事件流」（postgres=true / sqlite=false），
    前端据此出横幅说明「点进去为什么时间线是空的」（决定④）。
    """

    sessions: list[SessionInfo]
    history_available: bool = Field(
        default=True,
        description="事件流能否跨重启回放（PERSISTENCE_BACKEND=postgres 时为 true）",
    )


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
