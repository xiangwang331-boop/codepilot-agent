"""P7-2: PostgreSQL 连接池 seam 的单元测试（**不需要真的 PG**）。

覆盖：
- 池的连接参数必须与 `PostgresSaver.from_conn_string` 逐项一致（漏 autocommit = 事件静默全丢）
- 池必须显式 `open(wait=True, timeout=...)`，失败要关池 + 转 `PersistenceError`
- 建池时跑一次 checkpoint schema migration；`build_checkpointer(pool=...)` 不再跑 DDL
- `PostgresEventStore(pool=...)` 每条 SQL 借还连接，且**永不关共享池**
- `retry_cooldown`：长驻服务里 `_disabled` 不能是永久闩锁

真连 PG 的部分在 test_postgres_live.py（连不上自动 skip）。
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import Settings
from events.events import Event, EventType
from persistence import checkpointer as cp_mod
from persistence import event_store as es_mod
from persistence import pool as pool_mod
from persistence.checkpointer import PersistenceError, build_checkpointer
from persistence.event_store import PostgresEventStore
from persistence.pool import build_pool

# ---------------------------------------------------------------- 测试替身


class FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.calls.append((sql, params))
        if self._conn.should_fail(len(self._conn.calls)):
            raise RuntimeError(f"boom #{len(self._conn.calls)}")

    def fetchall(self):
        return []


class FakeConn:
    """假 psycopg 连接：记录每条 (sql, params)，可按调用序号注入失败。"""

    def __init__(self, *, should_fail=None):
        self.calls: list[tuple] = []
        self.should_fail = should_fail or (lambda n: False)
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True

    @property
    def inserts(self):
        return [(s, p) for s, p in self.calls if s.strip().upper().startswith("INSERT")]


class FakePool:
    """假连接池：记录借出次数，`close()` 只记一笔（不动底层连接）。"""

    def __init__(self, conn=None, *, fail_borrow=False):
        self.conn = conn or FakeConn()
        self.fail_borrow = fail_borrow
        self.borrows = 0
        self.closed = False

    @contextmanager
    def connection(self, timeout=None):
        self.borrows += 1
        if self.fail_borrow:
            raise RuntimeError("couldn't get a connection after 10.00 sec")
        yield self.conn

    def close(self):
        self.closed = True


def _patch_pool(monkeypatch, *, open_error: Exception | None = None):
    """把 pool_mod 的 ConnectionPool / PostgresSaver 换成记录型替身。

    返回 (建过的池, 建过的 saver)。
    """
    pools: list = []
    savers: list = []

    class RecPool:
        def __init__(self, conninfo, **kwargs):
            self.conninfo = conninfo
            self.kwargs = kwargs
            self.opened: list = []
            self.closed = False
            pools.append(self)

        def open(self, wait=False, timeout=30.0):
            if open_error is not None:
                raise open_error
            self.opened.append((wait, timeout))

        def close(self):
            self.closed = True

    class RecSaver:
        def __init__(self, conn, pipe=None, serde=None):
            self.conn = conn
            self.setups = 0
            savers.append(self)

        def setup(self):
            self.setups += 1

    monkeypatch.setattr(pool_mod, "ConnectionPool", RecPool)
    monkeypatch.setattr(pool_mod, "PostgresSaver", RecSaver)
    return pools, savers


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


_PG_URL = "postgresql://u:p@localhost:5432/db"


def _ev(message="m") -> Event:
    return Event(type=EventType.AGENT_STEP, agent="a", message=message, thread_id="t1")


# ---------------------------------------------------------------- 连接参数


def test_build_pool_connection_kwargs_match_saver_contract(monkeypatch, tmp_path):
    """池化版的「autocommit 必须为 True」。

    漏 autocommit → INSERT 停在隐式事务、连接归还池时既不提交也不报错 → **事件静默全丢**
    （池化后原样复现，最难查的一类 bug）。row_factory 缺了 → row["type"] 报 tuple indices。
    """
    pools, _ = _patch_pool(monkeypatch)
    build_pool(_settings(tmp_path, database_url=_PG_URL))

    (rec,) = pools
    conn_kwargs = rec.kwargs["kwargs"]
    assert conn_kwargs["autocommit"] is True
    assert conn_kwargs["prepare_threshold"] == 0
    assert conn_kwargs["row_factory"] is not None
    # ConnectionPool 的签名里没有 **kwargs，连接参数只能走 kwargs=
    assert "autocommit" not in rec.kwargs
    # 传的是副本：调用方拿不到模块常量去改
    assert conn_kwargs is not pool_mod.CONNECTION_KWARGS


def test_build_pool_opens_explicitly_with_wait(monkeypatch, tmp_path):
    """构造期自动 open 已废弃；且我们要求「库连不上就在启动时失败」而不是拖到首个会话。"""
    pools, _ = _patch_pool(monkeypatch)
    pool = build_pool(_settings(tmp_path, database_url=_PG_URL))

    (rec,) = pools
    assert rec is pool
    assert rec.kwargs["open"] is False
    assert rec.opened == [(True, pool_mod.POOL_OPEN_TIMEOUT)]
    assert rec.kwargs["min_size"] == pool_mod.POOL_MIN_SIZE
    assert rec.kwargs["max_size"] == pool_mod.POOL_MAX_SIZE


def test_build_pool_runs_checkpoint_migration_once(monkeypatch, tmp_path):
    """拿到 pool 就等于 schema 就绪 —— 所以 build_checkpointer(pool=...) 不再跑 DDL。"""
    pools, savers = _patch_pool(monkeypatch)
    pool = build_pool(_settings(tmp_path, database_url=_PG_URL))

    assert len(savers) == 1
    assert savers[0].setups == 1
    assert savers[0].conn is pool, "setup 必须发生在这份共享池上"


def test_build_pool_without_database_url_raises(tmp_path):
    with pytest.raises(PersistenceError, match="DATABASE_URL"):
        build_pool(_settings(tmp_path))


def test_build_pool_without_psycopg_pool_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(pool_mod, "ConnectionPool", None)
    with pytest.raises(PersistenceError, match="psycopg_pool"):
        build_pool(_settings(tmp_path, database_url=_PG_URL))


def test_build_pool_closes_pool_when_open_fails(monkeypatch, tmp_path):
    pools, _ = _patch_pool(monkeypatch, open_error=RuntimeError("boom"))
    with pytest.raises(PersistenceError):
        build_pool(_settings(tmp_path, database_url=_PG_URL))
    assert pools[0].closed, "启动失败必须关池，否则后台线程和连接一起泄漏"


def test_build_pool_pooltimeout_gets_actionable_hint(monkeypatch, tmp_path):
    class FakePoolTimeout(Exception):
        pass

    monkeypatch.setattr(pool_mod, "PoolTimeout", FakePoolTimeout)
    _patch_pool(monkeypatch, open_error=FakePoolTimeout("pool initialization incomplete"))
    with pytest.raises(PersistenceError, match="docker compose up -d"):
        build_pool(_settings(tmp_path, database_url=_PG_URL))


def test_build_pool_error_message_masks_password(monkeypatch, tmp_path):
    _patch_pool(monkeypatch, open_error=RuntimeError("boom"))
    with pytest.raises(PersistenceError) as ei:
        build_pool(_settings(tmp_path, database_url="postgresql://u:sup3rsecret@h:5432/db"))
    assert "sup3rsecret" not in str(ei.value)


# ---------------------------------------------------------------- checkpointer seam


def test_build_checkpointer_with_pool_skips_setup(monkeypatch, tmp_path):
    """传 pool = 调用方已建表；每会话再跑一遍 DDL 正是 P7 要消掉的开销。"""
    built: list = []

    class RecSaver:
        def __init__(self, conn, pipe=None, serde=None):
            self.conn = conn
            self.setups = 0
            built.append(self)

        def setup(self):
            self.setups += 1

    monkeypatch.setattr(cp_mod, "PostgresSaver", RecSaver)
    pool = object()

    with build_checkpointer(
        _settings(tmp_path, persistence_backend="postgres", database_url=_PG_URL),
        pool=pool,
    ) as saver:
        assert saver.conn is pool

    assert saver.setups == 0


def test_build_checkpointer_sqlite_ignores_pool(tmp_path):
    """sqlite 分支必须忽略 pool。

    sqlite 的正确共享方式是「全进程共用一个 SqliteSaver 实例」（多实例同文件会
    各自持锁 → database is locked 从 put 抛出来炸掉 graph run），跟 PG 池无关。
    """
    with build_checkpointer(_settings(tmp_path), pool=object()) as saver:
        assert isinstance(saver, SqliteSaver)


# ---------------------------------------------------------------- event store pool


def test_event_store_borrows_from_pool_and_never_closes_it():
    conn = FakeConn()
    pool = FakePool(conn)
    store = PostgresEventStore(_PG_URL, pool=pool)

    store.open()
    store.record(_ev())
    # 建表两条 + 插入一条，各借一次
    assert pool.borrows == 3
    assert "INSERT INTO events" in conn.inserts[0][0]

    store.close()
    assert pool.closed is False, "会话不得关掉共享池（关了之后所有会话的事件全丢）"
    assert conn.closed is False


def test_event_store_pool_open_is_idempotent():
    pool = FakePool()
    store = PostgresEventStore(_PG_URL, pool=pool)
    store.open()
    store.open()
    assert pool.borrows == 2, "第二次 open 不该重跑建表"


def test_event_store_record_never_raises_when_pool_unavailable(capsys):
    """record() 是 emitter 的 listener —— 池借不出连接也绝不能炸掉正在跑的 agent。"""
    pool = FakePool(fail_borrow=True)
    store = PostgresEventStore(_PG_URL, pool=pool)

    store.record(_ev())  # 不抛
    assert store.dropped_count == 1
    assert "事件持久化失败" in capsys.readouterr().err
    store.close()


def test_event_store_cooldown_reenables_writes(monkeypatch):
    """长驻服务里 _disabled 是永久闩锁的话，一次 PG 抖动 = 本进程余生都不落事件。"""
    state = {"fail": True, "now": 1000.0}
    conn = FakeConn(should_fail=lambda n: state["fail"] and n >= 3)  # 前 2 次是建表
    store = PostgresEventStore(
        _PG_URL, connect=lambda *a, **k: conn, retry_cooldown=30.0
    )
    monkeypatch.setattr(es_mod.time, "monotonic", lambda: state["now"])

    store.open()
    for _ in range(3):
        store.record(_ev())
    stalled = len(conn.inserts)
    assert stalled == 3

    # 冷却期内：继续丢弃，完全不碰连接
    state["now"] += 29.0
    store.record(_ev())
    assert len(conn.inserts) == stalled
    assert store.dropped_count == 4

    # 冷却到期 + 库回来 → 自动恢复写入
    state["now"] += 2.0
    state["fail"] = False
    store.record(_ev())
    assert len(conn.inserts) == stalled + 1
    assert store.dropped_count == 4  # 恢复了就不再丢
    store.close()


def test_event_store_without_cooldown_stays_latched(monkeypatch):
    """CLI 默认仍是 P6 的永久闩锁（进程很快就退，不值得重试）。"""
    state = {"now": 1000.0}
    conn = FakeConn(should_fail=lambda n: n >= 3)
    store = PostgresEventStore(_PG_URL, connect=lambda *a, **k: conn)
    monkeypatch.setattr(es_mod.time, "monotonic", lambda: state["now"])

    store.open()
    for _ in range(4):
        store.record(_ev())

    state["now"] += 10_000.0  # 时间过得再久也不恢复
    store.record(_ev())
    assert len(conn.inserts) == 3
    assert store.dropped_count == 5
    store.close()
