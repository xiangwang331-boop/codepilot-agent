"""P7: 应用组装 + lifespan —— 服务端进程级资源都在这一个地方建、一个地方收。

## 为什么没有模块级 `app = create_app()`

两个理由，都会疼：
1. `Settings.from_env()` 有副作用（往 `os.environ` 写 .env 的值），import 期就跑它是
   「import 一个模块顺手改了进程环境」——测试里 `monkeypatch` 环境变量也会被它抢先固化；
2. 测试要注入假 LLM（`make_llm`）和假容器（`runner_factory`），必须能带参数建 app。

所以起服务用 **factory** 形式：`uvicorn api.app:create_app --factory`。

## lifespan 里建什么、按什么顺序收

```
ExitStack:  pool ──► event_store ──► checkpointer(sqlite) ──► registry
                                                                 │
                  registry.__exit__ = stop_reaper + close_all ◄──┘  （先收会话）
                  …然后 checkpointer / event_store / pool 依次关闭
```

顺序是硬要求：会话握着容器与（postgres 下）从池里取的连接，必须**先于**池被关掉。
ExitStack 的 LIFO 天然满足——所以进栈顺序就是上面那样，不能改。

**sqlite 后端**：全进程**共用一个** `SqliteSaver` 实例（`SqliteSaver` 有
`check_same_thread=False` + 自带锁 + WAL，一个实例多线程共享是安全的；**多个**实例指向
同一文件才危险——各自一把锁，`database is locked` 会从 `put` 里抛出来炸掉 graph run）。
事件只在进程内存（`GET /events` 与 WS 回填读的就是它），重启即丢，启动时明确告警。

**postgres 后端**：全进程**共用一个连接池** + **每会话一个 `PostgresSaver(pool)`**。
每会话实例成本为零还消掉跨会话锁竞争；`setup()` 建表在 `build_pool()` 里跑过一次
（幂等），会话不再跑 DDL。
"""
from __future__ import annotations

import sys
import time
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api import routes, ws
from config.settings import Settings
from persistence.checkpointer import PersistenceError, build_checkpointer
from persistence.event_store import EventStoreError, PostgresEventStore
from persistence.pool import build_pool, close_pool
from runtime.registry import SessionRegistry
from tools.command_runner import sweep_orphan_containers

PROJECT_NAME = "CodePilot"
VERSION = "0.7.0"

# 静态测试页（P7-6）。挂载在最后：Starlette 按注册顺序匹配，先注册的 /sessions 优先，
# 剩下的才轮到 "/"。目录不存在就不挂（免得开发中间态起不来服务）。
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    settings: Settings | None = None,
    *,
    make_llm: Callable[[str], Any] | None = None,
    runner_factory: Callable[..., Any] | None = None,
    require_approval_for: tuple[str, ...] = ("coder",),
) -> FastAPI:
    """建一个 FastAPI 应用。

    `settings` 留空则 `Settings.from_env()`（**只调这一次**）。
    `make_llm` / `runner_factory` 是测试 seam：注入假 LLM 与假容器，
    传了 `make_llm` 就跳过 LLM 预热（测试不该等那 8.7 秒）。
    """
    settings = settings or Settings.from_env()
    warm_llm = make_llm is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if warm_llm:
            _warm_llm()

        _sweep_orphans(settings)

        with ExitStack() as stack:
            pool = _open_pool(settings, stack)
            event_store = _open_event_store(settings, stack, pool)
            checkpointer = _open_checkpointer(settings, stack)

            registry = stack.enter_context(
                SessionRegistry(
                    settings,
                    make_llm=make_llm,
                    runner_factory=runner_factory,
                    pool=pool,
                    event_store=event_store,
                    checkpointer=checkpointer,
                    require_approval_for=require_approval_for,
                    workspace_root=settings.workspace_root,
                    idle_timeout=settings.session_idle_timeout,
                )
            )
            registry.start_reaper()
            app.state.registry = registry
            _announce(settings, registry)
            try:
                yield
            finally:
                # 后面由 ExitStack 收：registry（会话 → 容器）→ checkpointer → store → pool
                app.state.registry = None

    app = FastAPI(title=PROJECT_NAME, version=VERSION, lifespan=lifespan)
    app.include_router(routes.router)
    app.include_router(ws.router)
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


# ---------------------------------------------------------------- lifespan 组件


def _sweep_orphans(settings: Settings) -> None:
    """启动时清掉上一个进程遗留的沙箱容器（见 `sweep_orphan_containers` 的前提说明）。

    只在 docker 模式且开关打开时跑。放在建 registry 之前：此刻还没有会话，
    任何带 label 的容器都只能是遗留物。
    """
    if settings.sandbox_mode != "docker":
        return
    if not settings.sweep_sandbox_on_start:
        print("提示：SWEEP_SANDBOX_ON_START=0 —— 跳过启动清扫，遗留容器不会被回收")
        return

    removed, error = sweep_orphan_containers()
    if error:
        # 清扫失败不挡启动：daemon 没起时后面建会话自然会给出更具体的报错。
        print(f"提示：启动清扫未完成（{error}）", file=sys.stderr)
    elif removed:
        print(f"启动清扫：回收了 {removed} 个遗留沙箱容器")


def _open_pool(settings: Settings, stack: ExitStack) -> Any | None:
    """postgres 后端的共享连接池；sqlite 后端返回 None。"""
    if settings.persistence_backend != "postgres":
        return None
    try:
        pool = build_pool(settings)
    except PersistenceError as e:
        _fatal("持久化初始化失败", e)
        raise
    stack.callback(close_pool, pool)
    return pool


def _open_event_store(
    settings: Settings, stack: ExitStack, pool: Any | None
) -> PostgresEventStore | None:
    """共享事件 store（只对 postgres 后端；sqlite 的事件只在内存里）。"""
    if settings.persistence_backend != "postgres":
        print(
            "提示：PERSISTENCE_BACKEND=sqlite —— 事件只存在进程内存里"
            "（GET /sessions/{id}/events 与 WS 回填读的就是它），**服务重启即丢**。\n"
            "      要持久化事件流：docker compose up -d 后设 PERSISTENCE_BACKEND=postgres",
            file=sys.stderr,
        )
        return None
    try:
        return stack.enter_context(
            PostgresEventStore(settings.database_url, pool=pool)
        )
    except EventStoreError as e:
        _fatal("事件 store 初始化失败", e)
        raise


def _open_checkpointer(settings: Settings, stack: ExitStack) -> Any | None:
    """sqlite → 返回**全进程共用**的那个 saver；postgres → None（每会话在池上自建）。

    共享实例由这里持有到服务退出；`build_runtime` 的「传进来的资产我不关」规则
    保证会话不会把它关掉（见 runtime/assembly.py 的所有权表）。
    """
    if settings.persistence_backend == "postgres":
        return None
    try:
        return stack.enter_context(build_checkpointer(settings))
    except PersistenceError as e:
        _fatal("checkpoint 初始化失败", e)
        raise


def _warm_llm() -> None:
    """预热 `langchain_openai` 的 import（实测 ≈8.7s）。

    真正的冷启动成本在 import，不在建图（建图 ~60ms）。放在 lifespan 里跑一次，
    否则第一个会话的第一条指令要莫名多等 8.7 秒。测试注入 make_llm 时跳过。
    """
    started = time.perf_counter()
    import langchain_openai  # noqa: F401  仅为预热 import

    print(f"预热：langchain_openai 已加载（{time.perf_counter() - started:.1f}s）")


def _announce(settings: Settings, registry: SessionRegistry) -> None:
    """启动横幅（可观测：一眼看清后端、沙箱模式、回收策略）。"""
    backend = (
        f"postgres（{_safe_url(settings.database_url)}）"
        if settings.persistence_backend == "postgres"
        else f"sqlite（{settings.checkpoint_db_path}，事件仅内存）"
    )
    sandbox = (
        f"docker（镜像 {settings.sandbox_image or 'codepilot-sandbox:py3.12'}，"
        "每会话一个容器，按 label codepilot.managed=1 标识）"
        if settings.sandbox_mode == "docker"
        else "local（run_command 跑在本机）"
    )
    print(
        f"{PROJECT_NAME} {VERSION} 服务已就绪\n"
        f"  持久化：{backend}\n"
        f"  沙箱  ：{sandbox}\n"
        f"  工作目录：{settings.workspace_root}（每会话一个子目录）\n"
        f"  空闲回收：{registry.idle_timeout:.0f}s 未活动（running 的会话永不回收）\n"
        f"  API   ：POST /sessions、POST /sessions/{{id}}/messages、"
        f"POST /sessions/{{id}}/approval、WS /sessions/{{id}}/ws\n"
        f"  测试页：http://127.0.0.1:8000/",
        flush=True,
    )


def _fatal(what: str, e: Exception) -> None:
    """启动期基础设施失败：给能照着做的提示，然后让 uvicorn 退出（不假装服务是好的）。"""
    print(f"\n错误：{what}。{e}", file=sys.stderr)


def _safe_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


__all__ = ["VERSION", "create_app"]
