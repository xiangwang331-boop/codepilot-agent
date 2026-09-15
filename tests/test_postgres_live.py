"""P6: PostgreSQL 集成测试 —— 真实 PG（模块级 skipif 自动跳过）。

刻意读 **TEST_DATABASE_URL** 而不是 DATABASE_URL：这些用例会真往库里写数据，
不该因为开发者本机 `.env` 设了 DATABASE_URL 就顺手污染他正在用的库。
起库：docker compose up -d
  $env:TEST_DATABASE_URL = "postgresql://codepilot:codepilot@localhost:5432/codepilot"

每个用例用自己的随机 thread_id 隔离，互不干扰、也无需清表
（events / checkpoints 都是按 thread_id 分区的）。
"""
from __future__ import annotations

import os
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph import build_agent_graph
from conftest import FakeLLM, _tool_call
from config.settings import Settings
from events.events import EventType, bind_thread, emit, emitter, reset_thread
from persistence.checkpointer import PersistenceError, build_checkpointer
from persistence.event_store import PostgresEventStore
from persistence.pool import build_pool, close_pool
from tools.registry import build_tools
from workspace.manager import WorkspaceManager

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


def _pg_ready() -> bool:
    """库可达才跑集成测试（连不上自动 skip，不拖慢主测试套件）。"""
    if not TEST_DATABASE_URL:
        return False
    try:
        import psycopg
    except ImportError:  # pragma: no cover  无 libpq 的环境
        return False
    try:
        with psycopg.connect(TEST_DATABASE_URL, connect_timeout=3):
            return True
    except Exception:  # noqa: BLE001  库没起 / 账号不对
        return False


pytestmark = pytest.mark.skipif(
    not _pg_ready(),
    reason="TEST_DATABASE_URL 未设置或 PostgreSQL 不可达（先 docker compose up -d）",
)


def _settings(tmp_path, **overrides) -> Settings:
    base = dict(
        llm_api_key="k",
        llm_base_url="",
        llm_model="m",
        workspace_root=tmp_path / "ws",
        checkpoint_db_path=tmp_path / "unused.db",  # postgres 后端下不使用
        persistence_backend="postgres",
        database_url=TEST_DATABASE_URL,
    )
    base.update(overrides)
    return Settings(**base)


def _initial(task="写文件"):
    return {
        "messages": [SystemMessage(content="s"), HumanMessage(content=task)],
        "current_agent": "Coder",
        "current_task": task,
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
        "tool_calls": [],
        "observations": [],
    }


def _write_script():
    return [
        _tool_call(1, "write_file", {"path": "a.py", "content": "x = 1\n"}),
        AIMessage(content="写好了"),
    ]


def _tid() -> str:
    return f"pg-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------- checkpoint


def test_pg_checkpoint_roundtrip_recovers_full_state(tmp_path):
    """对齐 test_persistence.py 的 sqlite 版：关库重开后状态必须完整恢复。"""
    tid = _tid()
    ws = WorkspaceManager(tmp_path / "ws")

    with build_checkpointer(_settings(tmp_path)) as saver:
        graph = build_agent_graph(FakeLLM(_write_script()), build_tools(ws), checkpointer=saver)
        result = graph.invoke(_initial(), {"configurable": {"thread_id": tid}})
        assert result["status"] == "finished"

    # 换一条连接重开（模拟进程重启）：checkpoint 必须从库里读回来
    with build_checkpointer(_settings(tmp_path)) as saver:
        graph = build_agent_graph(FakeLLM([]), build_tools(ws), checkpointer=saver)
        st = graph.get_state({"configurable": {"thread_id": tid}})
        msgs = st.values["messages"]
        # sys + human + tool_call + ToolMessage + final = 5
        assert len(msgs) == 5
        assert "写好了" in msgs[-1].content


def test_pg_resume_continues_on_recovered_history(tmp_path):
    tid = _tid()
    ws = WorkspaceManager(tmp_path / "ws")
    config = {"configurable": {"thread_id": tid}}

    with build_checkpointer(_settings(tmp_path)) as saver:
        graph = build_agent_graph(FakeLLM(_write_script()), build_tools(ws), checkpointer=saver)
        graph.invoke(_initial(), config)

    fake = FakeLLM([AIMessage(content="继续完成")])
    with build_checkpointer(_settings(tmp_path)) as saver:
        graph = build_agent_graph(fake, build_tools(ws), checkpointer=saver)
        result = graph.invoke({"messages": [HumanMessage(content="再写一个 b.py")]}, config)
        assert result["status"] == "finished"
        assert fake.calls == 1  # 基于恢复的历史继续决策，而不是重跑
        assert len(graph.get_state(config).values["messages"]) == 7


def test_pg_thread_isolation(tmp_path):
    """同库不同 thread 必须完全隔离 —— 这是 P7 多会话共用一个库的前提。"""
    t1, t2 = _tid(), _tid()
    ws = WorkspaceManager(tmp_path / "ws")

    with build_checkpointer(_settings(tmp_path)) as saver:
        graph = build_agent_graph(FakeLLM(_write_script()), build_tools(ws), checkpointer=saver)
        graph.invoke(_initial(), {"configurable": {"thread_id": t1}})

        assert len(graph.get_state({"configurable": {"thread_id": t2}}).values.get("messages", [])) == 0
        assert len(graph.get_state({"configurable": {"thread_id": t1}}).values["messages"]) == 5


def test_pg_setup_is_idempotent(tmp_path):
    """setup() 每次启动都调（main.py 的路径），必须可重复执行。"""
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(TEST_DATABASE_URL) as saver:
        saver.setup()
        saver.setup()  # 不抛 → checkpoint_migrations 跟踪生效

    # 连开两次 build_checkpointer（= 两次 setup）也不该出问题
    for _ in range(2):
        with build_checkpointer(_settings(tmp_path)) as s:
            assert s is not None


# ---------------------------------------------------------------- 事件落库


def test_pg_events_persist_ordered_and_isolated(tmp_path):
    """事件落库 → 回读：顺序正确、按 thread 隔离、detail/时间戳无损。"""
    tid_a, tid_b = _tid(), _tid()
    store = PostgresEventStore(TEST_DATABASE_URL)
    emitter.clear()
    try:
        store.open()
        assert store.dropped_count == 0
        emitter.add_listener(store.record)

        token = bind_thread(tid_a)
        try:
            emit(EventType.AGENT_STARTED, agent="Supervisor", message="")
            emit(EventType.AGENT_STEP, agent="Supervisor", message="m2", detail={"k": {"n": 1}})
            emit(EventType.TOOL_CALL_COMPLETED, agent="coder", message="m3")
        finally:
            reset_thread(token)

        token = bind_thread(tid_b)
        try:
            emit(EventType.AGENT_STEP, agent="Supervisor", message="other")
        finally:
            reset_thread(token)
    finally:
        emitter.clear()
        store.close()

    assert store.dropped_count == 0, "落库不该有降级"

    with PostgresEventStore(TEST_DATABASE_URL) as reader:
        a = reader.load(tid_a)
        b = reader.load(tid_b)

    assert [e.message for e in a] == ["", "m2", "m3"]  # 按写入顺序
    assert [e.type for e in a] == [
        EventType.AGENT_STARTED,
        EventType.AGENT_STEP,
        EventType.TOOL_CALL_COMPLETED,
    ]
    assert a[1].detail == {"k": {"n": 1}}  # jsonb 往返无损
    assert all(e.thread_id == tid_a for e in a)
    assert [e.message for e in b] == ["other"]

    # 时间戳以秒精度字符串往返（不静默腐化成 datetime 或丢时区）
    assert a[0].timestamp.endswith("+00:00") and len(a[0].timestamp) == 25


# ---------------------------------------------------------------- P7: 连接池


def test_build_pool_unreachable_database_is_friendly(tmp_path):
    """库不可达 → PersistenceError（带能照着做的提示），不是裸 psycopg 异常。"""
    settings = _settings(tmp_path, database_url="postgresql://nobody:nope@127.0.0.1:1/none")
    with pytest.raises(PersistenceError) as ei:
        build_pool(settings)
    message = str(ei.value)
    assert "Traceback" not in message
    assert "docker compose up -d" in message
    assert "nope" not in message, "报错信息里不能带密码"


def test_pool_backed_saver_roundtrip_and_reuse(tmp_path):
    """一份池 + 每会话一个 saver：跨会话复用同一池，状态各自完整。

    这正是 P7 服务端的持久化形态（见 persistence/pool.py 的说明）。
    """
    pool = build_pool(_settings(tmp_path))
    try:
        ws = WorkspaceManager(tmp_path / "ws")
        t1, t2 = _tid(), _tid()
        for tid in (t1, t2):
            with build_checkpointer(_settings(tmp_path), pool=pool) as saver:
                graph = build_agent_graph(
                    FakeLLM(_write_script()), build_tools(ws), checkpointer=saver
                )
                assert graph.invoke(_initial(), {"configurable": {"thread_id": tid}})[
                    "status"
                ] == "finished"

        # 换一个 saver 实例（同池）读回 t1：state 全在库里，与对象身份无关
        with build_checkpointer(_settings(tmp_path), pool=pool) as saver:
            graph = build_agent_graph(FakeLLM([]), build_tools(ws), checkpointer=saver)
            assert len(graph.get_state({"configurable": {"thread_id": t1}}).values["messages"]) == 5
            assert (
                len(graph.get_state({"configurable": {"thread_id": t2}}).values["messages"]) == 5
            )
    finally:
        close_pool(pool)


def test_pool_shares_safely_across_concurrent_sessions(tmp_path):
    """N 个会话线程共用一份池、各持一个 saver —— 互不串台、都不丢数据。

    PostgresSaver 自带 lock 只覆盖一次 DB 往返（不覆盖 LLM 调用），所以共享池是安全的；
    这条用例把「安全」钉成可执行的断言。
    """
    import threading

    pool = build_pool(_settings(tmp_path))
    ws = WorkspaceManager(tmp_path / "ws")
    tids = [_tid() for _ in range(4)]
    errors: list = []

    def worker(tid: str) -> None:
        try:
            with build_checkpointer(_settings(tmp_path), pool=pool) as saver:
                graph = build_agent_graph(
                    FakeLLM(_write_script()), build_tools(ws), checkpointer=saver
                )
                graph.invoke(_initial(), {"configurable": {"thread_id": tid}})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    try:
        threads = [threading.Thread(target=worker, args=(t,)) for t in tids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"并发会话不该抛异常: {errors}"

        with build_checkpointer(_settings(tmp_path), pool=pool) as saver:
            graph = build_agent_graph(FakeLLM([]), build_tools(ws), checkpointer=saver)
            for tid in tids:
                st = graph.get_state({"configurable": {"thread_id": tid}})
                assert len(st.values["messages"]) == 5, f"{tid} 的 checkpoint 不完整"
    finally:
        close_pool(pool)


def test_pool_backed_event_store_persists_and_never_closes_pool(tmp_path):
    """池化后 record() 仍「永不抛」且真落库；close() 不得关掉共享池。"""
    pool = build_pool(_settings(tmp_path))
    tid = _tid()
    store = PostgresEventStore(TEST_DATABASE_URL, pool=pool, retry_cooldown=30.0)
    emitter.clear()
    try:
        store.open()
        emitter.add_listener(store.record)
        token = bind_thread(tid)
        try:
            emit(EventType.AGENT_STEP, agent="a", message="pooled")
        finally:
            reset_thread(token)
    finally:
        emitter.clear()
        store.close()

    assert store.dropped_count == 0, "池化后不该降级丢事件"
    assert pool.closed is False, "store.close() 关掉共享池会让之后所有会话的事件全丢"

    with PostgresEventStore(TEST_DATABASE_URL, pool=pool) as reader:
        events = reader.load(tid)
    close_pool(pool)
    assert [e.message for e in events] == ["pooled"]
