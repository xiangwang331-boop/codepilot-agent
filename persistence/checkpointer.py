"""可插拔 checkpointer：SqliteSaver（默认）/ PostgresSaver（P6）。

P1 起 `build_agent_graph(llm, tools, checkpointer=None)` 就把 checkpointer 做成注入点，
这里只是把「选哪个 saver + 生命周期 + 建表」收敛成一个 context manager，图侧零改动。

用法：
    with build_checkpointer(settings) as checkpointer:
        graph = build_supervisor_graph(settings, ws, checkpointer=checkpointer)

两个后端的关键差异（已核对本地源码 langgraph/checkpoint/postgres/__init__.py:77,85）：
- `SqliteSaver` 会在首次 put 时**自动** setup；`PostgresSaver` **不会**，
  必须由调用方调一次 `setup()`（幂等，建表 + 跑 migration，用 checkpoint_migrations 表跟踪版本）。
- `PostgresSaver.from_conn_string()` 内部已设 `autocommit=True, prepare_threshold=0,
  row_factory=dict_row`（`CREATE INDEX CONCURRENTLY` 不能在事务块里跑，故必须 autocommit）。
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import Settings

# libpq 缺失（Windows 上没装 PostgreSQL 客户端工具链）时 import psycopg 会直接
# ImportError: no pq wrapper available。这里必须兜住 —— 否则连默认的 sqlite 后端
# 都会在 import 阶段就挂掉。同时保住了模块级名字，测试可 monkeypatch。
try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:  # pragma: no cover
    PostgresSaver = None  # type: ignore[assignment]


class PersistenceError(RuntimeError):
    """持久化基础设施问题（PG 连不上 / 缺 DATABASE_URL / 驱动缺失），不是 agent 代码的问题。"""


@contextmanager
def build_checkpointer(settings: Settings) -> Iterator[object]:
    """按 settings.persistence_backend 产出 checkpointer。

    sqlite（默认）→ data/checkpoints.db（会先建父目录）；
    postgres       → settings.database_url 指向的库（首次自动建表）。
    """
    if settings.persistence_backend == "postgres":
        with _postgres_checkpointer(settings) as saver:
            yield saver
    else:
        with _sqlite_checkpointer(settings) as saver:
            yield saver


@contextmanager
def _sqlite_checkpointer(settings: Settings) -> Iterator[object]:
    try:
        settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as saver:
            yield saver
    except OSError as e:
        raise PersistenceError(f"无法访问 SQLite 库 {settings.checkpoint_db_path}: {e}") from e


@contextmanager
def _postgres_checkpointer(settings: Settings) -> Iterator[object]:
    if PostgresSaver is None:
        raise PersistenceError(
            "postgres 后端需要 psycopg（含 libpq），当前环境不可用。\n"
            "  → 重装依赖: uv sync（pyproject 已声明 psycopg[binary]）"
        )
    if not settings.database_url:
        raise PersistenceError(
            "PERSISTENCE_BACKEND=postgres 需要 DATABASE_URL。\n"
            "  → 例: postgresql://codepilot:codepilot@localhost:5432/codepilot\n"
            "  → 起库: docker compose up -d"
        )

    # 注意 yield 必须放在 try **之外**：否则图体（正常任务执行）抛出的任何异常
    # 都会被下面这个 except 误包成 PersistenceError，掩盖真实错误。
    with ExitStack() as stack:
        try:
            saver = stack.enter_context(
                PostgresSaver.from_conn_string(settings.database_url)
            )
            # 幂等：SqliteSaver 自动 setup，PostgresSaver 必须显式调（否则 relation 不存在）。
            # 每次启动都调 = 每次启动都需要 DDL 权限；P7 服务化若要最小权限账号，
            # 在这里加一个跳过开关即可。
            saver.setup()
        except PersistenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise PersistenceError(_describe_pg_failure(e, settings.database_url)) from e
        yield saver


def _describe_pg_failure(e: Exception, database_url: str) -> str:
    """把 psycopg 异常翻译成能照着做的提示（连不上 / 没权限 是两类不同问题）。"""
    safe_url = database_url
    if "@" in safe_url:  # 别把密码打进日志
        scheme, _, rest = safe_url.partition("://")
        _, _, host = rest.rpartition("@")
        safe_url = f"{scheme}://***@{host}"

    if psycopg is not None and isinstance(e, psycopg.OperationalError):
        return (
            f"无法连接 PostgreSQL（{safe_url}）: {e}\n"
            "  → 库没起？ docker compose up -d\n"
            "  → 库不存在 / 账号密码或端口不对？ 检查 DATABASE_URL"
        )
    return (
        f"PostgreSQL 初始化失败（{safe_url}）: {type(e).__name__}: {e}\n"
        "  → checkpoint 表建不出来？ 该账号需要对目标 schema 的 CREATE 权限"
    )


def resolved_backend_label(settings: Settings) -> str:
    """给 CLI 打印用的一行描述（可观测性）。"""
    if settings.persistence_backend == "postgres":
        return f"PostgreSQL（{_safe(settings.database_url)}）"
    return f"SQLite（{settings.checkpoint_db_path}）"


def _safe(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


__all__ = [
    "PersistenceError",
    "build_checkpointer",
    "resolved_backend_label",
]
