"""P6: PostgreSQL 持久化的单元测试（**不需要真的 PG**）。

假的连接/假的 saver 覆盖：
- 事件打戳（thread_id 来自 ContextVar、step/node 来自图内 metadata）
- 事件行组装（Jsonb 包装、detail=None 存 SQL NULL）
- record() 的「永不抛异常」硬契约与降级语义
- 连接参数（autocommit / row_factory 缺一个就静默丢数据）
- build_checkpointer 的后端选择、setup() 调用、异常分类

真连 PG 的部分在 test_postgres_live.py（连不上自动 skip）。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import copy_context
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph import build_agent_graph
from conftest import FakeLLM, _tool_call
from config import settings as settings_mod
from config.settings import Settings
from events.events import (
    Event,
    EventType,
    bind_thread,
    current_thread_id,
    emitter,
    format_event,
    replay_stream,
    reset_thread,
)
from persistence import checkpointer as cp_mod
from persistence.checkpointer import PersistenceError, build_checkpointer
from persistence.event_store import (
    EventStoreError,
    PostgresEventStore,
    event_row,
    row_to_event,
)
from tools.registry import build_tools
from workspace.manager import WorkspaceManager

# ---------------------------------------------------------------- 测试替身


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._rows: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.calls.append((sql, params))
        if self._conn.should_fail(len(self._conn.calls)):
            raise RuntimeError(f"boom #{len(self._conn.calls)}")
        self._rows = list(self._conn.rows) if sql.strip().upper().startswith("SELECT") else []

    def fetchall(self):
        return self._rows


class FakeConn:
    """假 psycopg 连接：记录每条 (sql, params)，可按调用序号注入失败。"""

    def __init__(self, *, rows=(), should_fail=None):
        self.calls: list[tuple] = []
        self.rows = list(rows)
        self.should_fail = should_fail or (lambda n: False)
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True

    @property
    def inserts(self):
        return [(s, p) for s, p in self.calls if s.strip().upper().startswith("INSERT")]


def make_store(conn: FakeConn, url="postgresql://u:p@localhost:5432/db"):
    """store + 记录连接参数的假工厂。"""
    seen: dict = {}

    def factory(url_, **kwargs):
        seen["url"] = url_
        seen.update(kwargs)
        return conn

    return PostgresEventStore(url, connect=factory), seen


def make_fake_saver(raise_on_enter: Exception | None = None):
    """假的 PostgresSaver 类（供 monkeypatch.setattr 用）。"""

    class FakePostgresSaver:
        instances: list = []

        def __init__(self):
            self.setups = 0
            self.url = None

        def setup(self):
            self.setups += 1

        @classmethod
        @contextmanager
        def from_conn_string(cls, url, **kwargs):
            if raise_on_enter is not None:
                raise raise_on_enter
            saver = cls()
            saver.url = url
            cls.instances.append(saver)
            yield saver

    return FakePostgresSaver


def _settings(tmp_path, **overrides) -> Settings:
    base = dict(
        llm_api_key="k",
        llm_base_url="",
        llm_model="m",
        workspace_root=tmp_path / "ws",
        checkpoint_db_path=tmp_path / "data" / "checkpoints.db",
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


# ---------------------------------------------------------------- 事件打戳


def test_event_carries_bound_thread_id():
    emitter.clear()
    token = bind_thread("task-abc")
    try:
        e = emitter.emit(EventType.AGENT_STEP, agent="coder", message="x")
        assert e.thread_id == "task-abc"
    finally:
        reset_thread(token)
    assert current_thread_id() == ""

    # reset 之后发的事件不带会话归属
    e2 = emitter.emit(EventType.AGENT_STEP, agent="coder", message="y")
    assert e2.thread_id == ""
    emitter.clear()


def test_thread_binding_is_context_scoped():
    """ContextVar 语义：子上下文**继承**父上下文的绑定，但子上下文里的改写不回流。

    LangGraph 提交节点任务时正是 `copy_context()`（pregel/_executor.py:64），
    所以节点内看到 main.py 绑的会话 ID（继承方向，本方案赖以成立），
    而节点内若再 bind 也不会污染主线程（回流方向，P7 并发请求靠它互不串味）。
    """
    emitter.clear()
    token = bind_thread("mine")
    try:

        def child():
            assert current_thread_id() == "mine"  # 继承父上下文
            bind_thread("theirs")  # 子上下文内改写
            assert current_thread_id() == "theirs"

        copy_context().run(child)
        assert current_thread_id() == "mine"  # 未回流
    finally:
        reset_thread(token)
        emitter.clear()


def test_emit_captures_thread_id_and_step_inside_graph(tmp_path):
    """图内事件必须带上 thread_id + step/node —— 回放「↻ 重跑」判定全靠这两项。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    fake = FakeLLM(
        [
            _tool_call(1, "write_file", {"path": "a.py", "content": "x = 1\n"}),
            AIMessage(content="done"),
        ]
    )
    graph = build_agent_graph(fake, build_tools(ws))

    token = bind_thread("graph-thread")
    try:
        graph.invoke(_initial(), {"configurable": {"thread_id": "graph-thread"}})
    finally:
        reset_thread(token)

    tool_started = [e for e in emitter.events if e.type is EventType.TOOL_CALL_STARTED]
    assert tool_started, "应当发出 TOOL_CALL_STARTED"
    for e in tool_started:
        assert e.thread_id == "graph-thread"
        assert isinstance(e.step, int) and e.step >= 1
        assert e.node == "tools"
    # 图外事件没有位置信息
    outside = emitter.emit(EventType.AGENT_STARTED, agent="Supervisor", message="")
    assert outside.step is None and outside.node is None
    emitter.clear()


# ---------------------------------------------------------------- 回放判重


def _started(step, path, node="tools"):
    return Event(
        type=EventType.TOOL_CALL_STARTED,
        agent="Supervisor",
        message="read_file",
        detail={"args": {"path": path}},
        thread_id="t",
        step=step,
        node=node,
    )


def test_replay_marks_true_reruns_only():
    """同名工具的不同调用不能被误标「重跑」。

    TOOL_CALL_STARTED 的 message 只有工具名 —— 同一批次里 read_file a + read_file b
    两条 message 逐字相同。判重键只用 message 的话第二条会被误标（真机跑 P6 e2e 时
    实际出现过），键里带上 detail.args 才分得开。
    """
    out = replay_stream(
        [
            _started(2, "a.py"),
            _started(2, "b.py"),  # 同批的另一个调用：不是重跑
            _started(2, "a.py"),  # interrupt 恢复后节点从头执行：是重跑
        ]
    )
    assert [rerun for _, rerun in out] == [False, False, True]


def test_replay_never_marks_out_of_graph_events():
    """图外事件（AGENT_STARTED / 收尾的 AGENT_COMPLETED）没有 step，天然不判重。"""
    e = Event(type=EventType.AGENT_STARTED, agent="Supervisor", message="")
    assert [rerun for _, rerun in replay_stream([e, e, e])] == [False, False, False]


def test_replay_different_steps_are_not_reruns():
    """同一工具同参数在不同 superstep 出现 = 两次独立调用，不是重跑。"""
    out = replay_stream([_started(2, "a.py"), _started(4, "a.py")])
    assert [rerun for _, rerun in out] == [False, False]


# ---------------------------------------------------------------- 事件行组装


def test_event_row_wraps_detail_in_jsonb():
    row = event_row(
        Event(
            type=EventType.AGENT_STEP,
            agent="coder",
            message="m",
            detail={"args": {"path": "a.py"}},
            thread_id="t1",
            step=2,
            node="tools",
        )
    )
    assert row[0] == "t1" and row[1] == 2 and row[2] == "tools"
    assert row[3] == "AgentStep"  # 存枚举的 value 而不是 repr
    assert row[6] is not None and type(row[6]).__name__ == "Jsonb"


def test_event_row_none_detail_is_sql_null_not_json_null():
    """detail=None 要存 SQL NULL（而不是 JSON null）—— 两者语义不同。"""
    row = event_row(Event(type=EventType.AGENT_STEP, agent="a", message="m"))
    assert row[6] is None


def test_event_row_survives_unserializable_detail():
    """一条脏 detail 不该拖垮整条事件。"""
    row = event_row(
        Event(type=EventType.AGENT_STEP, agent="a", message="m", detail={"obj": object()})
    )
    assert row[6] is not None  # default=str 兜底，没炸


def test_row_to_event_roundtrip_renders_identically():
    """load 读回的 Event 经 format_event 渲染，必须与实时输出一致（回放契约）。"""
    original = Event(
        type=EventType.TOOL_CALL_COMPLETED,
        agent="coder",
        message="write_file(path='a.py')",
        detail={"k": 1},
        timestamp="2026-09-15T10:00:00+00:00",
        thread_id="t1",
        step=3,
        node="tools",
    )
    readback = row_to_event(
        {
            "thread_id": "t1",
            "step": 3,
            "node": "tools",
            "type": "ToolCallCompleted",
            "agent": "coder",
            "message": "write_file(path='a.py')",
            "detail": {"k": 1},
            "ts": datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc),
        }
    )
    assert format_event(readback) == format_event(original)
    assert readback.timestamp == original.timestamp  # datetime → 秒精度字符串，无类型腐化


def test_row_to_event_skips_unknown_type():
    assert row_to_event({"type": "SomethingNew", "agent": "a", "message": "m", "ts": None}) is None


# ---------------------------------------------------------------- 连接参数


def test_connect_uses_autocommit_and_dict_row():
    """缺 autocommit → INSERT 停在隐式事务里静默全丢；缺 dict_row → row['type'] 报错。

    这两项都不能靠「看着对」，必须在连接参数上钉死。
    """
    conn = FakeConn()
    store, seen = make_store(conn)
    store.open()
    assert seen["autocommit"] is True
    assert seen["row_factory"] is not None
    assert seen["connect_timeout"] == 5
    store.close()


def test_open_creates_schema_idempotently():
    conn = FakeConn()
    store, _ = make_store(conn)
    store.open()
    sqls = [s for s, _ in conn.calls]
    assert any("CREATE TABLE IF NOT EXISTS events" in s for s in sqls)
    assert any("CREATE INDEX IF NOT EXISTS" in s for s in sqls)
    assert any("thread_id, id" in s for s in sqls)  # 回放按 (thread_id, id) 走索引
    store.close()


def test_open_is_idempotent():
    conn = FakeConn()
    store, _ = make_store(conn)
    store.open()
    n = len(conn.calls)
    store.open()  # 第二次不该重连/重建表
    assert len(conn.calls) == n
    store.close()


def test_open_failure_raises_store_error():
    store, _ = make_store(FakeConn(should_fail=lambda n: True))
    with pytest.raises(EventStoreError):
        store.open()


# ---------------------------------------------------------------- record 契约


def test_record_inserts_row():
    conn = FakeConn()
    store, _ = make_store(conn)
    store.open()
    store.record(
        Event(
            type=EventType.TOOL_CALL_STARTED,
            agent="coder",
            message="write_file",
            detail={"args": {"path": "a.py"}},
            thread_id="t1",
            step=1,
            node="tools",
        )
    )
    (sql, params), = conn.inserts
    assert "INSERT INTO events" in sql
    assert params[0] == "t1" and params[1] == 1 and params[4] == "coder"
    store.close()


def test_record_never_raises_and_disables_after_threshold():
    """record() 是 EventEmitter 的 listener —— 抛异常会直接炸掉正在跑的 agent。"""
    conn = FakeConn(should_fail=lambda n: n >= 3)  # 前 2 次是建表，之后 INSERT 全失败
    store, _ = make_store(conn)
    store.open()

    for _ in range(4):
        store.record(Event(type=EventType.AGENT_STEP, agent="a", message="m"))  # 不抛

    assert store.dropped_count == 4
    # 前 3 条各试了一次都失败（拿到阈值即禁用），第 4 条根本没碰连接
    inserts = len(conn.inserts)
    assert inserts == 3
    store.record(Event(type=EventType.AGENT_STEP, agent="a", message="m"))
    assert len(conn.inserts) == inserts
    assert store.dropped_count == 5
    store.close()


def test_record_warns_once_then_goes_quiet(capsys):
    conn = FakeConn(should_fail=lambda n: n >= 3)
    store, _ = make_store(conn)
    store.open()
    for _ in range(2):
        store.record(Event(type=EventType.AGENT_STEP, agent="a", message="m"))
    err = capsys.readouterr().err
    assert err.count("事件持久化失败") == 1  # 只警一次，不刷屏
    store.close()


def test_record_recovers_after_transient_failure():
    """PG 重启这类瞬时故障不该被当成永久故障一次性判死。"""
    state = {"fail": True}
    conn = FakeConn(should_fail=lambda n: n >= 3 and state["fail"])
    store, _ = make_store(conn)
    store.open()

    store.record(Event(type=EventType.AGENT_STEP, agent="a", message="m1"))
    assert store.dropped_count == 1

    state["fail"] = False  # 库回来了
    store.record(Event(type=EventType.AGENT_STEP, agent="a", message="m2"))
    store.record(Event(type=EventType.AGENT_STEP, agent="a", message="m3"))
    assert store.dropped_count == 1  # 后续都写进去了
    assert len(conn.inserts) == 3
    store.close()


def test_record_opens_lazily_when_not_opened():
    conn = FakeConn()
    store, _ = make_store(conn)
    store.record(Event(type=EventType.AGENT_STEP, agent="a", message="m"))
    assert len(conn.inserts) == 1
    store.close()


def test_close_is_idempotent():
    conn = FakeConn()
    store, _ = make_store(conn)
    store.open()
    store.close()
    store.close()  # atexit 兜底也会调，必须幂等
    assert conn.closed


# ---------------------------------------------------------------- load


def test_load_reconstructs_ordered_events():
    rows = [
        {
            "thread_id": "t1",
            "step": 1,
            "node": "agent",
            "type": "AgentStep",
            "agent": "Supervisor",
            "message": "决定调用 1 个工具",
            "detail": None,
            "ts": datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc),
        },
        {
            "thread_id": "t1",
            "step": 2,
            "node": "tools",
            "type": "ToolCallCompleted",
            "agent": "Supervisor",
            "message": "delegate(...)",
            "detail": {"args": {"specialist": "coder"}},
            "ts": datetime(2026, 9, 15, 10, 0, 1, tzinfo=timezone.utc),
        },
    ]
    conn = FakeConn(rows=rows)
    store, _ = make_store(conn)
    events = store.load("t1")

    assert [e.type for e in events] == [EventType.AGENT_STEP, EventType.TOOL_CALL_COMPLETED]
    assert [e.step for e in events] == [1, 2]
    assert events[1].detail == {"args": {"specialist": "coder"}}
    select_sql, params = conn.calls[-1]
    assert "ORDER BY id" in select_sql and params == ("t1",)
    store.close()


def test_load_warns_and_skips_unknown_types(capsys):
    rows = [
        {"thread_id": "t", "step": 1, "node": "n", "type": "Nope", "agent": "a", "message": "m", "detail": None, "ts": None},
        {"thread_id": "t", "step": 1, "node": "n", "type": "AgentStep", "agent": "a", "message": "ok", "detail": None, "ts": None},
    ]
    store, _ = make_store(FakeConn(rows=rows))
    events = store.load("t")
    assert len(events) == 1 and events[0].message == "ok"
    assert "类型未知" in capsys.readouterr().err
    store.close()


# ---------------------------------------------------------------- build_checkpointer


def test_build_checkpointer_sqlite_is_default_and_creates_parent_dir(tmp_path):
    db = tmp_path / "nested" / "data" / "checkpoints.db"
    settings = _settings(tmp_path, checkpoint_db_path=db)
    assert settings.persistence_backend == "sqlite"

    with build_checkpointer(settings) as saver:
        assert saver is not None
        assert type(saver).__name__ == "SqliteSaver"  # 真 saver，不是替身
    assert db.parent.is_dir()  # 父目录由 build_checkpointer 负责建（原先在 main.py）


def test_build_checkpointer_postgres_calls_setup_once(tmp_path, monkeypatch):
    fake_cls = make_fake_saver()
    monkeypatch.setattr(cp_mod, "PostgresSaver", fake_cls)

    settings = _settings(
        tmp_path, persistence_backend="postgres", database_url="postgresql://u:p@h:5432/db"
    )
    with build_checkpointer(settings) as saver:
        assert saver.url == "postgresql://u:p@h:5432/db"  # URL 原样透传
    assert len(fake_cls.instances) == 1
    # SqliteSaver 会自动 setup，PostgresSaver 不会 —— 不调就是 relation does not exist
    assert fake_cls.instances[0].setups == 1


def test_build_checkpointer_postgres_requires_database_url(tmp_path, monkeypatch):
    monkeypatch.setattr(cp_mod, "PostgresSaver", make_fake_saver())
    settings = _settings(tmp_path, persistence_backend="postgres", database_url="")
    with pytest.raises(PersistenceError, match="DATABASE_URL"):
        with build_checkpointer(settings):
            pass


def test_build_checkpointer_wraps_connection_error(tmp_path, monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    monkeypatch.setattr(
        cp_mod, "PostgresSaver", make_fake_saver(raise_on_enter=psycopg.OperationalError("nope"))
    )
    settings = _settings(
        tmp_path, persistence_backend="postgres", database_url="postgresql://u:secret@h:5432/db"
    )
    with pytest.raises(PersistenceError) as ei:
        with build_checkpointer(settings):
            pass
    assert "docker compose up -d" in str(ei.value)
    assert "secret" not in str(ei.value)  # 密码不能进报错信息


def test_build_checkpointer_does_not_mask_body_exceptions(tmp_path, monkeypatch):
    """图体（正常任务执行）抛出的异常必须原样冒泡，不能被误包成 PersistenceError。

    这是 _postgres_checkpointer 里把 yield 放在 try 之外的原因。
    """
    monkeypatch.setattr(cp_mod, "PostgresSaver", make_fake_saver())
    settings = _settings(
        tmp_path, persistence_backend="postgres", database_url="postgresql://u:p@h:5432/db"
    )
    with pytest.raises(ValueError, match="agent boom"):
        with build_checkpointer(settings):
            raise ValueError("agent boom")


def test_missing_psycopg_gives_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cp_mod, "PostgresSaver", None)  # 模拟 libpq 缺失环境
    settings = _settings(
        tmp_path, persistence_backend="postgres", database_url="postgresql://u:p@h:5432/db"
    )
    with pytest.raises(PersistenceError, match="psycopg"):
        with build_checkpointer(settings):
            pass


# ---------------------------------------------------------------- 配置


def test_persistence_backend_defaults_to_sqlite(monkeypatch):
    # 屏蔽 .env 加载：开发者本机的 .env 可能设了 PERSISTENCE_BACKEND=postgres，
    # 那是本机偏好，不该决定解析逻辑的测试结果。
    monkeypatch.setattr(settings_mod, "_load_dotenv", lambda: None)
    monkeypatch.delenv("PERSISTENCE_BACKEND", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings.from_env()
    assert s.persistence_backend == "sqlite"
    assert s.database_url == ""


def test_persistence_backend_invalid_falls_back_with_warning(monkeypatch, capsys):
    monkeypatch.setattr(settings_mod, "_load_dotenv", lambda: None)
    monkeypatch.setenv("PERSISTENCE_BACKEND", "mysql")
    s = Settings.from_env()
    assert s.persistence_backend == "sqlite"
    assert "非法" in capsys.readouterr().err


def test_persistence_backend_postgres_and_url(monkeypatch):
    monkeypatch.setattr(settings_mod, "_load_dotenv", lambda: None)
    monkeypatch.setenv("PERSISTENCE_BACKEND", "POSTGRES")  # 大小写归一
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/db")
    s = Settings.from_env()
    assert s.persistence_backend == "postgres"
    assert s.database_url == "postgresql://u:p@h:5432/db"


# ---------------------------------------------------------------- 端到端（图 → store）

def test_graph_events_land_in_store_with_thread_id(tmp_path):
    """绑定会话 → 跑真实图 → store 收到的事件全部带该 thread_id。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    fake = FakeLLM(
        [
            _tool_call(1, "write_file", {"path": "a.py", "content": "x = 1\n"}),
            AIMessage(content="done"),
        ]
    )
    graph = build_agent_graph(fake, build_tools(ws))

    conn = FakeConn()
    store, _ = make_store(conn)
    store.open()
    emitter.add_listener(store.record)

    token = bind_thread("e2e-thread")
    try:
        graph.invoke(_initial(), {"configurable": {"thread_id": "e2e-thread"}})
    finally:
        reset_thread(token)
        store.close()
        emitter.clear()

    assert conn.inserts, "事件应当被写库"
    assert all(p[0] == "e2e-thread" for _, p in conn.inserts), "每条事件都该带会话 ID"
    assert (tmp_path / "ws" / "a.py").exists()  # 顺带确认图真的跑了
