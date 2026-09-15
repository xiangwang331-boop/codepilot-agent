"""P7-1: per-session emitter 的单元测试（不需要网络、不需要容器、不需要 PG）。

覆盖：
- 未绑定 ContextVar 时 emit() 回退模块单例（CLI 行为零变化的锚点）
- 绑定的会话 emitter 收全图内事件，模块单例保持干净
- 多线程各绑各的 emitter，互不串台
- 「必须改 emit() 函数体、不能靠 monkeypatch 模块属性」这条约束

**为什么不用 fixture**：沿用项目惯例，直接吃 pytest 的 tmp_path。
"""
from __future__ import annotations

import threading

from langchain_core.messages import AIMessage, SystemMessage

from agent.graph import build_agent_graph
from conftest import FakeLLM, _tool_call
from events.events import (
    EventEmitter,
    EventType,
    bind_emitter,
    current_emitter,
    emit,
    emitter,
    reset_emitter,
)
from tools.registry import build_tools
from workspace.manager import WorkspaceManager


def _initial() -> dict:
    """与 main.py / test_postgres.py 一致的初始 state。"""
    return {
        "messages": [SystemMessage(content="sys"), AIMessage(content="hi")],
        "current_agent": "Coder",
        "current_task": "写个文件",
        "iteration_count": 0,
        "max_iterations": 20,
        "status": "running",
        "tool_calls": [],
        "observations": [],
    }


def _run_graph(ws_root, script, thread_id: str) -> None:
    """跑一个最小 ReAct 闭环（写文件 + 收尾），调用方负责 bind_emitter。"""
    ws = WorkspaceManager(ws_root)
    graph = build_agent_graph(FakeLLM(script), build_tools(ws))
    graph.invoke(_initial(), {"configurable": {"thread_id": thread_id}})


# ---------------------------------------------------------------- 回退语义


def test_unbound_emit_falls_back_to_module_singleton():
    """未绑定时走模块单例 —— 这是「CLI 行为零变化」的锚点。"""
    emitter.clear()
    try:
        assert current_emitter() is emitter
        e = emit(EventType.AGENT_STARTED, agent="Supervisor", message="")
        assert e in emitter.events
    finally:
        emitter.clear()


def test_reset_emitter_restores_fallback():
    """reset 之后回到模块单例（worker 线程收尾必须 reset，避免污染复用线程）。"""
    emitter.clear()
    try:
        own = EventEmitter()
        token = bind_emitter(own)
        assert current_emitter() is own
        reset_emitter(token)
        assert current_emitter() is emitter
    finally:
        emitter.clear()


# ---------------------------------------------------------------- 图内生效


def test_bound_emitter_receives_graph_events_and_singleton_stays_empty(tmp_path):
    """绑定后图内事件进会话 emitter，模块单例一条都不该有。"""
    emitter.clear()
    own = EventEmitter()
    token = bind_emitter(own)
    try:
        _run_graph(
            tmp_path / "ws",
            [
                _tool_call(1, "write_file", {"path": "a.py", "content": "x = 1\n"}),
                AIMessage(content="done"),
            ],
            "sess-a",
        )
    finally:
        reset_emitter(token)

    try:
        types = [e.type for e in own.events]
        assert EventType.TOOL_CALL_STARTED in types, "会话 emitter 应当收到工具事件"
        assert EventType.TOOL_CALL_COMPLETED in types
        # 事件归属也应当是对的
        assert all(e.thread_id == "sess-a" for e in own.events if e.thread_id)
        # 模块单例必须干净 —— 否则 CLI 那个单例会看到服务端会话的事件
        assert emitter.events == []
    finally:
        emitter.clear()


def test_bound_emitter_does_not_leak_after_reset(tmp_path):
    """reset 之后图内事件回到模块单例，会话 emitter 不再增长。"""
    emitter.clear()
    own = EventEmitter()
    token = bind_emitter(own)
    _run_graph(tmp_path / "ws", [AIMessage(content="done")], "sess-b")
    reset_emitter(token)

    n_after_bound = len(own.events)
    _run_graph(tmp_path / "ws", [AIMessage(content="done")], "sess-b")

    try:
        assert len(own.events) == n_after_bound, "reset 后不该再往会话 emitter 写"
        assert emitter.events, "reset 后应当回落到模块单例"
    finally:
        emitter.clear()


# ---------------------------------------------------------------- 并发隔离


def test_emitters_isolated_across_threads(tmp_path):
    """多个会话各绑各的 emitter，事件不串台（P7 并发的核心保证）。"""
    emitter.clear()
    n = 6
    collected: dict[str, list] = {}
    errors: list = []

    def worker(i: int) -> None:
        own = EventEmitter()
        collected[f"s{i}"] = own
        token = bind_emitter(own)
        try:
            _run_graph(
                tmp_path / f"ws{i}",
                [
                    _tool_call(1, "write_file", {"path": f"f{i}.py", "content": "x=1\n"}),
                    AIMessage(content=f"done-{i}"),
                ],
                f"s{i}",
            )
        except Exception as e:  # noqa: BLE001  收进列表，别让线程里的异常静默丢失
            errors.append(e)
        finally:
            reset_emitter(token)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert not errors, f"工作线程不应抛异常: {errors}"
        assert len(collected) == n
        for tid, own in collected.items():
            assert own.events, f"{tid} 应当收到事件"
            # 每个 emitter 只该有自己会话的事件
            foreign = [e for e in own.events if e.thread_id and e.thread_id != tid]
            assert not foreign, f"{tid} 的 emitter 混入了别的会话事件: {foreign}"
        # 模块单例全程未被写入
        assert emitter.events == [], "并发会话不该写进 CLI 的模块单例"
    finally:
        emitter.clear()


# ---------------------------------------------------------------- 约束钉死


def test_module_level_emit_must_be_changed_in_body_not_patched(monkeypatch):
    """把「per-session emitter 只能靠改 emit() 函数体」这条约束钉死。

    agent/core.py / agent/condense.py / agent/supervisor.py 都是
    `from events.events import emit`——import 时就把**函数对象**绑进了各自的命名空间。
    所以 monkeypatch.setattr(events, "emit", fake) 对它们毫无影响。
    这正是 P7 的实现方式（改函数体）而非替换模块属性的原因；
    哪天有人「优化」成 monkeypatch 方案，这条测试会立刻失败。
    """
    from agent import core
    from events import events

    assert core.emit is events.emit, "core 绑的是同一个函数对象"

    sentinel = lambda *a, **k: None  # noqa: E731
    monkeypatch.setattr(events, "emit", sentinel)

    assert events.emit is sentinel
    assert core.emit is not sentinel, (
        "core.emit 不该被 monkeypatch 影响——若这条挂了，说明 import 方式变了，"
        "per-session emitter 的实现方式需要重新评估"
    )
