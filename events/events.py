"""事件系统。

从第一版就记录事件，未来 Web UI 直接消费同一套事件流。
P0 只做线性日志（刻意舍弃 OpenHands 的事件树 parent_id 分支）。

P6 起事件带「会话归属」与「图内位置」两组元数据，供落库与回放：
- thread_id : 由 ContextVar 提供，`main.py`（未来是 FastAPI 请求）在 invoke 前
  `bind_thread()` 一次即可，**core/condense/supervisor 的 emit 调用点零改动**。
  注意不要改成从 `ensure_config()` 读 thread_id——specialist 子图用的是
  `delegate-N` 这个内部 thread_id，那样一个会话的事件会被拆进多个桶。
- step/node: 从 `ensure_config()` 的 metadata 读当前 LangGraph 节点位置。
  resume 时被 interrupt 打断的节点会**从头重跑**，重跑步的 step/node 与首次完全相同，
  因此回放侧可用 (step, node, type, message) 识别并标注「↻ 重跑」。
"""
from __future__ import annotations

import json
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable

from langchain_core.runnables.config import ensure_config

# P6: 当前会话 ID。用 ContextVar 而非全局变量，是为了 P7 的并发请求能各自绑定；
# LangGraph 提交节点任务时会 copy_context()（见 pregel/_executor.py），
# 所以在 main.py 里 set 的值在节点内可见（已探针实证）。
_current_thread: ContextVar[str] = ContextVar("codepilot_thread_id", default="")


def bind_thread(thread_id: str) -> Token:
    """把当前上下文绑定到某个会话 ID，返回 token 供 reset_thread 还原。"""
    return _current_thread.set(thread_id)


def reset_thread(token: Token) -> None:
    """还原 bind_thread 之前的会话 ID。"""
    _current_thread.reset(token)


def current_thread_id() -> str:
    return _current_thread.get()


def current_graph_position() -> tuple[int | None, str | None]:
    """读当前 LangGraph 节点位置 (langgraph_step, langgraph_node)。

    图外（如 main.py 收尾发的 AGENT_COMPLETED）没有该 metadata，返回 (None, None)。
    刻意吞异常：emit 在每个节点里都会调，这里出问题会炸掉整个 agent。
    """
    try:
        metadata = ensure_config().get("metadata") or {}
    except Exception:  # noqa: BLE001
        return None, None
    return metadata.get("langgraph_step"), metadata.get("langgraph_node")


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
    # 用户下发的指令本身。**只有服务端发**（`runtime/session.py` 的 worker 起步处），
    # 图内不发 —— 所以 CLI（不走 `Session`）的事件流与输出逐字不变。
    #
    # 为什么它必须是一条事件而不是「前端自己记一下」：指令是这一轮所有 agent 事件的
    # **前置语境**，只有进了事件流才拿得到正确的 seq（于是时间线上它就落在它引发的那轮
    # 之前）、才落得进 PG（于是刷新/重启/换设备都还在）。前端本地记一份在 F5 之后就没了。
    USER_MESSAGE = "UserMessage"


@dataclass
class Event:
    type: EventType
    agent: str
    message: str
    detail: dict[str, Any] | None = None
    timestamp: str = ""
    # P6: 会话归属 + 图内位置（落库/回放用；图外事件 thread_id 可能为空、step/node 为 None）
    thread_id: str = ""
    step: int | None = None
    node: str | None = None

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
        step, node = current_graph_position()
        e = Event(
            type=type,
            agent=agent,
            message=message,
            detail=detail,
            thread_id=current_thread_id(),
            step=step,
            node=node,
        )
        self.events.append(e)
        for fn in self._listeners:
            fn(e)
        return e

    def clear(self) -> None:
        """重置整个 emitter（事件 + 监听器）。

        监听器也要清：否则测试里挂的监听器（如事件落库）会活到整个 pytest 进程结束，
        静默影响后续所有测试。生产侧 main.py 只在启动时挂一次、且从不调 clear。
        """
        self.events.clear()
        self._listeners.clear()


# 模块级单例：CLI（单进程单会话）用；未绑定 ContextVar 时 emit() 回退到这里。
emitter = EventEmitter()

# P7: 当前上下文的 emitter。与 _current_thread 同理——用 ContextVar 而非全局变量，
# 让每个会话的 worker 线程各自绑定，于是**该会话图内所有 emit 自动落到自己的 emitter**。
# 机制同 P6 已验证的 bind_thread：LangGraph 提交节点任务时 copy_context()。
_current_emitter: ContextVar["EventEmitter | None"] = ContextVar(
    "codepilot_emitter", default=None
)


def bind_emitter(em: EventEmitter) -> Token:
    """把当前上下文绑定到某个会话专属 emitter，返回 token 供 reset_emitter 还原。

    必须在**每个 worker 线程启动时**调用（含审批恢复后新起的线程）——这是线程
    上下文作用域，不是会话作用域，线程不会继承别的线程的绑定。
    """
    return _current_emitter.set(em)


def reset_emitter(token: Token) -> None:
    """还原 bind_emitter 之前的 emitter。"""
    _current_emitter.reset(token)


def current_emitter() -> EventEmitter:
    """当前上下文的 emitter；未绑定时回退模块单例（CLI 行为零变化）。

    ⚠️ 服务端代码请用 `session.emitter.emit(...)`，**不要**调模块级 `emit()`：
    event loop 线程没有绑定任何会话 emitter，回退到模块单例后事件会进 CLI 那个
    全局 emitter，会话订阅者永远收不到（且没有任何报错）。
    """
    return _current_emitter.get() or emitter


def emit(type: EventType, agent: str, message: str, detail: dict | None = None) -> Event:
    # P7: 走 current_emitter() 而非直接 emitter —— 这一行改动即让 per-session
    # emitter 全量生效。**必须改函数体**：agent/core.py、agent/condense.py、
    # agent/supervisor.py 都是 `from events.events import emit`，import 时就把这个
    # 函数对象绑进了各自的命名空间，monkeypatch.setattr(events.events, "emit", ...)
    # 对它们无效（有单测把这条约束钉死）。
    return current_emitter().emit(type, agent, message, detail)


# 回放时打在重跑事件行尾的标记（main.py 与测试共用，避免文案漂移）
RERUN_MARK = "   ↻ 重跑（interrupt 恢复后节点从头执行）"


def _replay_key(e: Event) -> tuple:
    """回放判重键。必须带上 detail，不能只用 message：
    `TOOL_CALL_STARTED` 的 message 只有工具名，同一批里连调两次同名工具
    （read_file a + read_file b）会撞成同一个键、第二次被误标成「重跑」——真机实测踩过。
    带上 detail（TOOL_CALL_STARTED 的 detail 是 {"args": ...}）即可区分，
    而真正的重跑参数逐字相同，仍能被识别。

    已知边界：同一批里用**完全相同的参数**连调同一个工具两次，第二次会被误标。
    极少见，且标注只是提示性的。
    """
    detail = "" if e.detail is None else json.dumps(e.detail, sort_keys=True, default=str)
    return (e.step, e.node, getattr(e.type, "value", e.type), e.message, detail)


def replay_stream(events: Iterable[Event]) -> list[tuple[str, bool]]:
    """把一段事件流渲染成 [(行, 是否重跑)]，供 `--events` 回放。

    resume 时被 interrupt 打断的节点会从头重跑，同一条事件因此落库两次
    （step/node/message/detail 逐字相同，已探针实证）——这里识别并标记。
    图外事件（AGENT_STARTED 等）没有 step，天然不参与判重。
    """
    seen: set[tuple] = set()
    out: list[tuple[str, bool]] = []
    for e in events:
        line = format_event(e)
        rerun = False
        if e.step is not None:
            key = _replay_key(e)
            rerun = key in seen
            seen.add(key)
        out.append((line, rerun))
    return out


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
    if e.type is EventType.USER_MESSAGE:
        # 需求可以是多行的（Web 的输入框支持 Shift+Enter）——按 Condense 摘要的同一
        # 手法逐行加前缀，别让第二行起裸奔、看起来像别的 agent 的输出。
        head, *rest = e.message.splitlines() or [""]
        out = f"=== 需求 ===\n{head}"
        return out + "".join("\n" + line for line in rest)
    return f"{prefix} {e.message}"
