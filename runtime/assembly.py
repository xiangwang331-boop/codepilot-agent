"""P7: 会话装配层 —— CLI（main.py）与 API（runtime/session.py）共用。

**这一层只造对象，不打印、不发事件。** CLI 的输出字节不变靠这条保证（打印留在
main.py），服务端的事件归属靠调用方绑定（见 events.bind_emitter）。

两边共用的不只是装配，还有 interrupt 循环（`runtime/driver.py`）——装配漂移只是
「CLI 能跑、服务不能跑」，而挂起语义漂移会让同一个审批在两条路径下行为不同，更严重。

## 所有权规则：谁造谁收

| 参数 | 传进来 | 没传 |
|---|---|---|
| `checkpointer` | 调用方的资产，这里不关 | 按 settings 造，退出时关 |
| `event_store` | 调用方的资产，这里不关（服务端是全进程共享的那个） | postgres 后端才造，退出时关 |
| `runner` | 调用方的资产，这里**不启动** | 按 settings 造，**仍不启动** |

`runner` 一律不在这里启动：CLI 要在**容器起来之前**先打印「沙箱模式：…」那一行
（容器起不来时那行是用户唯一的线索），服务端则由会话注册表按时机起停。
调用方用 `rt.sandbox()` 进/出，或自己直接 `with` 那个 runner 对象。
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from agent.condense import recursion_limit_for
from agent.specialists import build_supervisor_prompt
from agent.supervisor import build_supervisor_graph
from config.settings import Settings
from events.events import EventEmitter
from events.events import emitter as default_emitter
from persistence.checkpointer import build_checkpointer
from persistence.event_store import PostgresEventStore
from tools.command_runner import DEFAULT_IMAGE, DockerCommandRunner
from workspace.manager import WorkspaceManager


@dataclass
class SessionRuntime:
    """一个会话的全部装配件。生命周期由 `build_runtime` 的 with 块界定。"""

    thread_id: str
    settings: Settings
    ws: WorkspaceManager
    graph: Any
    config: dict
    emitter: EventEmitter
    runner: Any | None = None  # 未启动；调用方用 sandbox() 或自己 with
    event_store: PostgresEventStore | None = None

    def initial_input(self, task: str) -> dict:
        """全新任务的初始 state（与 P0 起的 AgentState 字段一一对应）。"""
        return {
            "messages": [
                SystemMessage(content=build_supervisor_prompt()),
                HumanMessage(content=task),
            ],
            "current_agent": "Supervisor",
            "current_task": task,
            "iteration_count": 0,
            "max_iterations": self.settings.max_iterations,
            "status": "running",
            "tool_calls": [],
            "observations": [],
        }

    def append_input(self, task: str) -> dict:
        """恢复会话时追加一条新指令。

        **只带这一条 HumanMessage** —— 重传初始 messages 会被 add_messages 追加成重复历史。
        """
        return {"messages": [HumanMessage(content=task)]}

    def existing_message_count(self) -> int:
        """会话已有多少条消息（0 = 这个会话不存在）。"""
        return len(self.graph.get_state(self.config).values.get("messages", []))

    @contextmanager
    def sandbox(self) -> Iterator[None]:
        """起停沙箱容器；local 模式是空包（行为与 P0–P4 一致）。"""
        with (self.runner or nullcontext()):
            yield


@contextmanager
def build_runtime(
    settings: Settings,
    *,
    thread_id: str,
    workspace_root: Path | str,
    make_llm: Callable[[str], Any] | None = None,
    runner_factory: Callable[[Settings, WorkspaceManager], Any] | None = None,
    runner: Any | None = None,
    require_approval_for: tuple[str, ...] = ("coder",),
    checkpointer: Any | None = None,
    pool: Any | None = None,
    event_store: PostgresEventStore | None = None,
    emitter: EventEmitter | None = None,
) -> Iterator[SessionRuntime]:
    """装配一个会话；退出时收掉自己造的东西（见模块 docstring 的所有权表）。

    `make_llm` / `runner_factory` 是测试 seam（分别对齐 P2 的 make_llm 与 P5 的
    假 runner 注入），生产不传。
    """
    em = emitter if emitter is not None else default_emitter
    ws = WorkspaceManager(Path(workspace_root))
    if runner is None:
        runner = _default_runner(settings, ws, runner_factory)

    with ExitStack() as stack:
        if checkpointer is None:
            checkpointer = stack.enter_context(build_checkpointer(settings, pool=pool))

        if event_store is None and settings.persistence_backend == "postgres":
            # sqlite 后端保持 P0–P5 行为：事件只在进程内存里，不落库
            event_store = stack.enter_context(
                PostgresEventStore(settings.database_url, pool=pool)
            )
        if event_store is not None:
            em.add_listener(event_store.record)

        graph = build_supervisor_graph(
            settings,
            ws,
            make_llm=make_llm,
            checkpointer=checkpointer,
            require_approval_for=require_approval_for,
            command_runner=runner,
        )

        yield SessionRuntime(
            thread_id=thread_id,
            settings=settings,
            ws=ws,
            graph=graph,
            config={
                "configurable": {"thread_id": thread_id},
                # P4-3: supervisor 是 agent→tools→condense 三节点循环，按 max_iterations
                # 放大 recursion_limit，保证 max_iterations 才是真正的循环上限
                # （LangGraph 默认 25 只够 ~8 轮）。
                "recursion_limit": recursion_limit_for(settings.max_iterations),
            },
            emitter=em,
            runner=runner,
            event_store=event_store,
        )


def _default_runner(
    settings: Settings,
    ws: WorkspaceManager,
    runner_factory: Callable[[Settings, WorkspaceManager], Any] | None,
) -> Any | None:
    """按 settings 决定 run_command 的执行宿主（P5）。"""
    if runner_factory is not None:
        return runner_factory(settings, ws)
    if settings.sandbox_mode == "docker":
        return DockerCommandRunner(ws.root, image=settings.sandbox_image or DEFAULT_IMAGE)
    return None  # local：run_command 走本机 subprocess


__all__ = ["SessionRuntime", "build_runtime"]
