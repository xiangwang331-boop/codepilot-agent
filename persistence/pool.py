"""P7: PostgreSQL 连接池 —— 全进程共享一份。

CLI 的用法**完全不变**（`build_checkpointer(settings)` 内部自己建连、with 退出即弃）。
服务端在 lifespan 里调一次 `build_pool(settings)`，之后每个会话只在这份池上建自己的
`PostgresSaver(pool)` —— P6 里每会话现连现断 + 每会话跑一遍 `setup()` DDL 的代价就没了。

**为什么是「一份池 + 每会话一个 saver」而不是「一份池 + 一份 saver」**：
`PostgresSaver` 自带 `self.lock`，`_cursor()` 只在**一次 DB 往返**期间持锁（已核对源码：
`with self.lock, _internal.get_connection(self.conn) as conn`），所以共享实例也不会串台，
但每会话一个实例成本为零、又能顺带消掉跨会话的锁竞争 —— 反正 ExitStack 本来就要每会话建。

**连接参数必须与 `PostgresSaver.from_conn_string` 逐项一致**（见 checkpointer.py 注释）：
- `autocommit=True` 缺了 → INSERT 停在隐式事务、连接归还池时既不提交也不报错 → **事件静默全丢**
  （P6 关键坑 #22 在池化后原样复现，是最难查的一类 bug）
- `prepare_threshold=0` → 关掉 prepared statement 复用
- `row_factory=dict_row` 缺了 → `row["type"]` 报 tuple indices

注意 psycopg_pool 的连接参数走 **`kwargs={...}`**，不是 `**kwargs` ——
构造签名里没有 `**kwargs`，直接展开会 TypeError。
"""
from __future__ import annotations

from typing import Any

from config.logging_setup import get_logger
from config.settings import Settings
from persistence.checkpointer import PersistenceError, PostgresSaver, _safe

logger = get_logger(__name__)

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool, PoolTimeout
except ImportError:  # pragma: no cover
    # 没 libpq 的 Windows 上，sqlite 模式必须还能 import 本模块（同 P6 的守卫方式）。
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    ConnectionPool = None  # type: ignore[assignment]
    PoolTimeout = None  # type: ignore[assignment]


# 单机开发够用，也远低于 PG 默认 max_connections=100；并发会话多了再调。
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 8
POOL_ACQUIRE_TIMEOUT = 10.0  # 单次借连接最多等多久（默认 30s 太长，会话会干等）
POOL_OPEN_TIMEOUT = 5.0  # 启动时等首批连接的上限

CONNECTION_KWARGS: dict[str, Any] = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}


def build_pool(settings: Settings) -> Any:
    """建共享连接池并打开。失败抛 `PersistenceError`（友好提示，不打 traceback）。

    **成功返回即意味着 checkpoint schema 已就绪**（见 `_ensure_checkpoint_schema`）——
    所以拿到 pool 的调用方不必（也不该）再跑 `setup()`。
    """
    if ConnectionPool is None:
        raise PersistenceError(
            "postgres 后端需要 psycopg_pool（含 psycopg/libpq），当前环境不可用。\n"
            "  → 重装依赖: uv sync（pyproject 已声明 psycopg-pool）"
        )
    if not settings.database_url:
        raise PersistenceError(
            "PERSISTENCE_BACKEND=postgres 需要 DATABASE_URL。\n"
            "  → 例: postgresql://codepilot:codepilot@localhost:5432/codepilot\n"
            "  → 起库: docker compose up -d"
        )

    pool = ConnectionPool(
        settings.database_url,
        min_size=POOL_MIN_SIZE,
        max_size=POOL_MAX_SIZE,
        timeout=POOL_ACQUIRE_TIMEOUT,
        open=False,  # 显式 open：构造期自动 open 已废弃，且我们要「连不上就在启动时炸」
        kwargs=dict(CONNECTION_KWARGS),
    )
    try:
        # wait=True：等到 min_size 条连接就绪才返回。库不可达在这里就报错，
        # 而不是拖到第一个会话跑起来才在 worker 线程里炸 —— 那时用户已经在等结果了。
        pool.open(wait=True, timeout=POOL_OPEN_TIMEOUT)
        _ensure_checkpoint_schema(pool)
    except Exception as e:  # noqa: BLE001
        pool.close()  # 幂等；wait 超时时 psycopg_pool 内部已经 close 过一次
        if isinstance(e, PersistenceError):
            raise
        raise PersistenceError(_describe_pool_failure(e, settings.database_url)) from e
    return pool


def close_pool(pool: Any | None) -> None:
    """幂等关闭。**只由池的属主调用**（应用 shutdown）—— 会话不得自行关掉共享池。"""
    if pool is None:
        return
    try:
        pool.close()
    except Exception as e:  # noqa: BLE001  关闭失败不致命（进程即将退出）
        logger.warning("警告: 关闭连接池失败: %s", e, exc_info=True)


def _ensure_checkpoint_schema(pool: Any) -> None:
    """借一个临时 saver 跑一次幂等 migration。

    `setup()` 与实例无关（建表 + 版本表 + migration，见 checkpointer.py 的说明），
    所以借实例在**建池时跑一次**就够了，每会话不再跑 DDL ——
    顺带把 P6 注释里记的那笔 defer（「每次启动都要求 DDL 权限」）收掉。
    """
    if PostgresSaver is None:  # pragma: no cover  上面 ConnectionPool 可用时它必然可用
        raise PersistenceError(
            "postgres 后端需要 psycopg（含 libpq），当前环境不可用 → uv sync"
        )
    PostgresSaver(pool).setup()


def _describe_pool_failure(e: Exception, database_url: str) -> str:
    """把 psycopg/psycopg_pool 异常翻译成能照着做的提示。"""
    safe = _safe(database_url)
    if PoolTimeout is not None and isinstance(e, PoolTimeout):
        return (
            f"连接池在 {POOL_OPEN_TIMEOUT:.0f}s 内没能建起连接（{safe}）: {e}\n"
            "  → 库没起？ docker compose up -d\n"
            "  → 账号密码或端口不对？ 检查 DATABASE_URL"
        )
    if psycopg is not None and isinstance(e, psycopg.OperationalError):
        return (
            f"无法连接 PostgreSQL（{safe}）: {e}\n"
            "  → 库没起？ docker compose up -d\n"
            "  → 库不存在 / 账号密码或端口不对？ 检查 DATABASE_URL"
        )
    return (
        f"PostgreSQL 连接池初始化失败（{safe}）: {type(e).__name__}: {e}\n"
        "  → 建 checkpoint 表失败？ 该账号需要对目标 schema 的 CREATE 权限"
    )


__all__ = [
    "CONNECTION_KWARGS",
    "POOL_ACQUIRE_TIMEOUT",
    "POOL_MAX_SIZE",
    "POOL_MIN_SIZE",
    "POOL_OPEN_TIMEOUT",
    "build_pool",
    "close_pool",
]
