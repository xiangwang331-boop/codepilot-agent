"""事件落库 + 回放读取（P6）。

`PostgresEventStore` 通过 `emitter.add_listener(store.record)` 挂上事件流——
所以 agent/core.py、agent/condense.py、agent/supervisor.py 的 emit 调用点**零改动**。

两条容易踩的坑，都在这里兜住：

1. **连接必须 `autocommit=True`**。psycopg3 默认把每条 INSERT 停在隐式事务里，
   连接关闭时既不提交也不报错 —— 事件会静默全丢。（`PostgresSaver.from_conn_string`
   内部帮自己设了 autocommit，但我们这条连接得自己设。）
2. **`record()` 必须永不抛异常**。`EventEmitter.emit`（events/events.py）是裸的
   `for fn in self._listeners: fn(e)`，没有 try —— listener 抛异常会直接冒泡进图节点
   炸掉 agent（TOOL_CALL_STARTED 的 emit 就在 core.py 的 try 之外）。事件是可观测性，
   不该杀死正在跑的任务：失败一律降级为告警 + 丢弃，连续失败到阈值就停写。
"""
from __future__ import annotations

import atexit
import json
import sys
from datetime import datetime
from typing import Any, Callable

from events.events import Event, EventType

# psycopg 缺失时（无 libpq 的 Windows）本模块仍可 import，
# 只有真正用 postgres 后端时才会在 open() 报错。
try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    Jsonb = None  # type: ignore[assignment]

# step/node 记录事件发生在哪个 LangGraph 节点、哪个 superstep。
# resume 时被 interrupt 打断的节点会从头重跑，重跑步的 step/node 与首次**完全相同**
# （已探针实证），因此回放侧靠 (step, node, type, message) 就能识别「↻ 重跑」。
EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id        BIGSERIAL   PRIMARY KEY,
    thread_id TEXT        NOT NULL,
    step      INTEGER,
    node      TEXT,
    type      TEXT        NOT NULL,
    agent     TEXT        NOT NULL,
    message   TEXT        NOT NULL,
    detail    JSONB,
    ts        TIMESTAMPTZ NOT NULL
)
"""

# 排序用全局单调的 BIGSERIAL id：同 thread 内 ORDER BY id 即正确时序，
# 避开 per-thread「MAX(seq)+1」在并发下的竞态。
EVENTS_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS events_thread_id_id_idx ON events (thread_id, id)"
)

_INSERT_SQL = (
    "INSERT INTO events (thread_id, step, node, type, agent, message, detail, ts)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
)

_SELECT_SQL = (
    "SELECT thread_id, step, node, type, agent, message, detail, ts"
    " FROM events WHERE thread_id = %s ORDER BY id"
)

# 连续失败到几次才彻底停写：PG 重启这类瞬时故障不该被当成永久故障一次性判死。
_FAILURE_THRESHOLD = 3


class EventStoreError(RuntimeError):
    """事件存储的基础设施问题（连不上库 / 建不出表 / 驱动缺失）。"""


def _jsonb(value: Any):
    """psycopg3 不会自动把 dict 适配成 jsonb，必须用 Jsonb 包一层。

    detail=None 要存 SQL NULL 而不是 JSON null —— 两者语义不同，read 侧需能区分。
    非 JSON 安全的值用 default=str 兜底，不让一条脏 detail 拖垮整条事件。
    """
    if value is None:
        return None
    return Jsonb(value, dumps=_dumps) if Jsonb is not None else _dumps(value)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def event_row(event: Event) -> tuple:
    """Event → INSERT 参数元组（纯函数，单测直接断言）。"""
    return (
        event.thread_id,
        event.step,
        event.node,
        getattr(event.type, "value", str(event.type)),
        event.agent,
        event.message,
        _jsonb(event.detail),
        event.timestamp,
    )


def row_to_event(row: dict) -> Event | None:
    """DB 行 → Event；type 不认识（表被更新版本写过）时返回 None，由调用方计数。

    ts 读回来是 datetime（列是 TIMESTAMPTZ），而 Event.timestamp 是秒精度字符串 ——
    转回同一格式，避免静默的类型腐化。
    """
    try:
        etype = EventType(row["type"])
    except ValueError:
        return None
    ts = row.get("ts")
    return Event(
        type=etype,
        agent=row.get("agent") or "",
        message=row.get("message") or "",
        detail=row.get("detail"),
        timestamp=ts.isoformat(timespec="seconds") if isinstance(ts, datetime) else str(ts or ""),
        thread_id=row.get("thread_id") or "",
        step=row.get("step"),
        node=row.get("node"),
    )


class PostgresEventStore:
    """事件落 PostgreSQL 的写入器 + 回放读取器。

    生命周期与 checkpointer **分开管理**：checkpointer 的 with 块在 main.py 里
    结束于结果打印之前，而收尾的 AGENT_COMPLETED / AGENT_FAILED 事件发在那之后 ——
    绑在一起每次会话的最后一条事件必丢。这里用 `open()`/`close()` + atexit 兜底。
    """

    def __init__(self, database_url: str, *, connect: Callable | None = None):
        self.database_url = database_url
        self._connect_factory = connect  # 测试 seam（默认 psycopg.connect）
        self._conn = None
        self._disabled = False
        self._failures = 0
        self._dropped = 0
        self._warned = False
        atexit.register(self.close)

    # ---- 可替换 seam（单测用，对应 P5 DockerCommandRunner._run_cli 的思路）----

    def _connect(self):
        if psycopg is None:
            raise EventStoreError(
                "事件落库需要 psycopg（含 libpq），当前环境不可用 → uv sync"
            )
        factory = self._connect_factory or psycopg.connect
        return factory(
            self.database_url,
            autocommit=True,  # 必须：否则 INSERT 停在隐式事务里，静默全丢
            row_factory=dict_row,  # 必须：否则 row["type"] 报 tuple indices
            connect_timeout=5,
        )

    def _execute(self, sql: str, params: tuple | None = None) -> None:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def _query(self, sql: str, params: tuple | None = None) -> list:
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # ---- 生命周期 ----

    def open(self) -> "PostgresEventStore":
        """建连 + 建表（幂等）。失败抛 EventStoreError。"""
        if self._conn is not None:
            return self
        try:
            self._conn = self._connect()
            self._execute(EVENTS_DDL)
            self._execute(EVENTS_INDEX_DDL)
        except Exception as e:  # noqa: BLE001
            self._conn = None
            if isinstance(e, EventStoreError):
                raise
            raise EventStoreError(_describe_failure(e, self.database_url)) from e
        return self

    def close(self) -> None:
        """幂等关闭（atexit 兜底也会调）。"""
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001  关闭失败静默（进程将退出）
            pass
        finally:
            self._conn = None

    def __enter__(self) -> "PostgresEventStore":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- 写 ----

    @property
    def dropped_count(self) -> int:
        """本次运行因持久化不可用而丢弃的事件数（收尾打印用）。"""
        return self._dropped

    def record(self, event: Event) -> None:
        """把一条事件写库。**契约：永不抛异常**（见模块 docstring）。"""
        if self._disabled:
            self._dropped += 1
            return
        try:
            if self._conn is None:
                self.open()
            self._execute(_INSERT_SQL, event_row(event))
            self._failures = 0
        except Exception as e:  # noqa: BLE001
            self._failures += 1
            self._dropped += 1
            if not self._warned:
                self._warned = True
                print(
                    f"警告: 事件持久化失败，已降级（事件仍在本进程内存中）: {e}",
                    file=sys.stderr,
                )
            if self._failures >= _FAILURE_THRESHOLD:
                self._disabled = True
                print(
                    f"警告: 事件持久化连续失败 {self._failures} 次，本次运行不再尝试写入。",
                    file=sys.stderr,
                )

    # ---- 读（回放）----

    def load(self, thread_id: str, *, limit: int | None = None) -> list[Event]:
        """按写入顺序读回一个会话的事件流（回放用）。"""
        if self._conn is None:
            self.open()
        sql = _SELECT_SQL + (" LIMIT %s" if limit is not None else "")
        params = (thread_id, limit) if limit is not None else (thread_id,)
        rows = self._query(sql, params)

        events: list[Event] = []
        skipped = 0
        for row in rows:
            e = row_to_event(row)
            if e is None:
                skipped += 1
                continue
            events.append(e)
        if skipped:
            print(
                f"警告: {skipped} 条事件类型未知（表可能由更新版本写入），已跳过。",
                file=sys.stderr,
            )
        return events


def _describe_failure(e: Exception, database_url: str) -> str:
    safe = _safe_url(database_url)
    if psycopg is not None and isinstance(e, psycopg.OperationalError):
        return (
            f"无法连接 PostgreSQL（{safe}）: {e}\n"
            "  → 库没起？ docker compose up -d\n"
            "  → 账号密码或端口不对？ 检查 DATABASE_URL"
        )
    return f"事件表初始化失败（{safe}）: {type(e).__name__}: {e}"


def _safe_url(url: str) -> str:
    """打日志别把密码带出去。"""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


__all__ = [
    "EVENTS_DDL",
    "EVENTS_INDEX_DDL",
    "EventStoreError",
    "PostgresEventStore",
    "event_row",
    "row_to_event",
]
