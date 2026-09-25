"""P7: 应用组装 + lifespan —— 服务端进程级资源都在这一个地方建、一个地方收。

## 为什么没有模块级 `app = create_app()`

两个理由，都会疼：
1. `Settings.from_env()` 有副作用（往 `os.environ` 写 .env 的值），import 期就跑它是
   「import 一个模块顺手改了进程环境」——测试里 `monkeypatch` 环境变量也会被它抢先固化；
2. 测试要注入假 LLM（`make_llm`）和假容器（`runner_factory`），必须能带参数建 app。

所以起服务用 **factory** 形式：`uvicorn api.app:create_app --factory`。

## lifespan 里建什么、按什么顺序收

```
ExitStack:  pool ──► event_store ──► checkpointer(sqlite) ──► catalog ──► registry
                                                                            │
                  registry.__exit__ = stop_reaper + close_all  ◄────────────┘
                  …然后 catalog ──► checkpointer ──► event_store ──► pool 依次关闭
```

顺序是硬要求：会话握着容器与（postgres 下）从池里取的连接，必须**先于**池被关掉；
catalog 的探针图也读 checkpointer/池，所以要**晚于**它们进栈、**早于** registry 出栈。
ExitStack 的 LIFO 天然满足——所以进栈顺序就是上面那样，不能改。

**sqlite 后端**：全进程**共用一个** `SqliteSaver` 实例（`SqliteSaver` 有
`check_same_thread=False` + 自带锁 + WAL，一个实例多线程共享是安全的；**多个**实例指向
同一文件才危险——各自一把锁，`database is locked` 会从 `put` 里抛出来炸掉 graph run）。
**共享实例必须同时交给 registry 与 catalog**（`checkpointer=` 参数）——它们各建一个
指向同一文件的实例就正是上面那个危险情形。事件只在进程内存（`GET /events` 与 WS 回填
读的就是它），重启即丢，启动时明确告警。

**postgres 后端**：全进程**共用一个连接池** + **每会话一个 `PostgresSaver(pool)`**。
每会话实例成本为零还消掉跨会话锁竞争；`setup()` 建表在 `build_pool()` 里跑过一次
（幂等），会话不再跑 DDL。

## P9：catalog —— 读路径的接驳层

`SessionCatalog`（`runtime/catalog.py`）在启动时枚举持久化里的会话、从 checkpoint 重推
各自状态，交给 `registry.discover()` 登记成**未物化记录**：`GET /sessions` 立刻有内容，
点开某个历史会话时 registry 才把它物化成真 `Session`。没有这一步，重启后
「库里有、列表里没有」——那正是 P9 要修的那个问题。

它**不参与写路径**：会话照旧只由 `POST /sessions` 建。
"""
from __future__ import annotations

import os
import time
from contextlib import ExitStack, asynccontextmanager
from os import PathLike
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, Response
from starlette.types import Scope

from api import routes, ws
from config.logging_setup import get_logger, setup_logging, teardown_logging
from config.settings import Settings
from persistence.checkpointer import PersistenceError, build_checkpointer
from persistence.event_store import EventStoreError, PostgresEventStore
from persistence.pool import build_pool, close_pool
from runtime.catalog import SessionCatalog
from runtime.registry import SessionRegistry
from tools.command_runner import sweep_orphan_containers

PROJECT_NAME = "CodePilot"
VERSION = "0.7.0"

logger = get_logger(__name__)

# 静态测试页（P7-6）。挂载在最后：Starlette 按注册顺序匹配，先注册的 /sessions 优先，
# 剩下的才轮到 "/"。目录不存在就不挂（免得开发中间态起不来服务）。
STATIC_DIR = Path(__file__).resolve().parent / "static"


class _HashedAssetStaticFiles(StaticFiles):
    """静态托管 + 缓存策略：入口 HTML 永远整份重取，带哈希的产物长缓存。

    为什么非加不可（对着真机实测踩到）：`api/static/` 的入口 HTML 是一份**资源清单** ——
    它引用的是带内容哈希的文件名，`npm run build` 每跑一次哈希就变一次。而 StaticFiles
    默认**只发 `etag`/`last-modified`、不发 `Cache-Control`**，浏览器于是按启发式规则
    （约「距 last-modified 时长的 10%」）自行决定复用、**根本不回源校验**。

    后果是重建之后用户刷新页面，拿到的仍是缓存里的旧 `index.html`，而它指向的旧 JS/CSS
    也在缓存里 → **整套界面退回改动前的样子**：实测症状正是「委派卡被压成 2px + 时间线
    滚不动」，与服务端磁盘上已经是新的完全无关。这类「明明修好了却还是老样子」最难自查，
    因为它看起来像修复没生效。

    策略分两半，依据是「文件名会不会随内容变」：
      · `*.html`（其实就是 index.html）→ `no-store`，**并且不发任何验证器**
        （`etag` / `last-modified` 一起摘掉）→ 浏览器既不存它、也没有东西可以拿去做条件
        请求，于是**每一次进入都必须整份重取**（本机 427 字节，代价可忽略）；
      · 其余（Vite 产物，名字里带内容哈希）→ 长缓存 + `immutable`：内容变了名字必然变，
        老名字永远不会被请求第二次，缓存越久越好。

    ## 为什么从 `no-cache` 退到 `no-store`（2026-09-25 晚，实测后改的）

    原先写的是 `no-cache`（**可以**存、但每次必须回源校验，etag 命中就是 304，代价极小）。
    那条推理本身没错，错在它默认了一件事：**浏览器会把 304 响应里的 `cache-control`
    合并回那份已存的副本**。而一份副本的新鲜度是**用它自己那份响应头**判定的 —— 那份头是
    和副本一起存下来的，不是服务器现在会发的那组。于是「已经被存下来的旧副本」是一个
    服务端再也够不着的状态，能不能自愈全押在上面那条合并行为上；一旦不成立（或那份副本的
    启发式新鲜期还没走完），表现就是 **「每次进去都是旧的、点一下刷新才对」** —— 正是用户
    反复报的现象，而且刷新一次并不保证就此终结。

    `no-store` + 摘掉验证器把这条依赖整个删掉：**没有存得下的副本，也就没有「旧副本」这个
    状态**。代价是每次进入多付一次 427 字节的整份重取（放弃了 304 这条更便宜的路），换来
    这个状态不可能再出现。本机工具，这个交换很划算。

    ## 为什么还要发 `Clear-Site-Data: "cache"`（2026-09-25 深夜·四，实测后加）

    ⛔ **上面那句「旧副本这个状态不可能再出现」只对「将来存下的副本」成立 ——
    它对**已经存在浏览器里**的那份副本毫无作用。** 用户随后那句
    「一进去必须刷新一次、关掉浏览器再进去还是要刷新」（**每次**，永不自愈）就是这半句话的
    反例，而它**不是缓存没配好**，是**缓存键**的问题：

    - HTTP 缓存键是**完整 URL**，而 `App.tsx` 选中会话用的是 `history.replaceState`
      —— **不产生导航**，只是把地址栏改成 `/?session=…`，文档还是从 `/` 加载的那一份；
    - 于是用户真实走法是：**进 `/`**（命中改 `no-store` 之前存下的旧副本 → **零请求**、
      旧界面）→ 点会话（无导航，仍是那个旧文档）→ **在 `/?session=…` 上刷新**
      （**另一个缓存键**，那里没有副本 → 回源 → 拿到新的 → 「刷新就好了」）；
    - **`/` 那份旧副本从头到尾没被碰过。** 关掉浏览器再进 `/` → 又是它。
      循环**永不自愈**，与「重建了几次」无关。

    所以在**交付出文档**（每次真回源的那一刻）顺带叫浏览器把本站缓存清掉：旧副本被删，
    而 `no-store` 保证不再存新的 → **清一次、永久生效**。这是服务端唯一还能碰到客户端
    那份副本的时机 —— 缓存命中时它连请求都不发，任何头都到不了它手上。

    实测（玩具服务端 + 持久化 profile，跑用户的完整循环，各臂独立 profile）：
    不发这个头的臂在「关浏览器 → 重开 → 进 `/`」拿到**旧界面且服务端 0 个请求**；
    发这个头的臂拿到**新界面**，且 `/` 与产物都被重新取过。

    代价（明确接受）：它也清掉本站的**产物缓存**，于是 `immutable` 那份长缓存每进一次页面
    就作废一次、产物重新取一遍。本机回环、产物就那几个文件，换来的是「一进去就是对的」。
    **只发给文档，绝不给产物** —— 给产物会让每次资源加载都清一次缓存，那是病态的。
    """

    def file_response(
        self, full_path: PathLike, stat_result: os.stat_result, scope: Scope, status_code: int = 200
    ) -> Response:
        if Path(full_path).suffix == ".html":
            # 自己建响应、**不走 super()**：304 的判定是在 super() 内部做完的
            # （`starlette/staticfiles.py` 的 `file_response` 末尾调 `is_not_modified`，
            # 拿响应上的 etag 与请求的 `If-None-Match` 比），事后再删头已经晚了 ——
            # 必须从一开始就不给它验证器，304 这条路径根本不存在。
            response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
            response.headers["cache-control"] = "no-store"
            del response.headers["etag"]
            del response.headers["last-modified"]
            # 清掉浏览器里可能已经存下的旧副本 —— 唯一能追溯作废它的手段。
            response.headers["clear-site-data"] = '"cache"'
            return response
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["cache-control"] = "public, max-age=31536000, immutable"
        return response


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
        # ⚠️ **必须是第一句**：下面 `_warm_llm`（要 8.7 秒）与 `_sweep_orphans` 的输出
        # 都该带上新格式、也都该进日志文件。放晚了这两条就是最后一批「裸 print」。
        # 它自己**绝不抛**（内部包了 try/except），所以这里不用管它的失败。
        setup_logging(settings)
        try:
            if warm_llm:
                _warm_llm()

            _sweep_orphans(settings)

            with ExitStack() as stack:
                pool = _open_pool(settings, stack)
                event_store = _open_event_store(settings, stack, pool)
                checkpointer = _open_checkpointer(settings, stack)
                catalog = _open_catalog(
                    settings, stack, pool, event_store, checkpointer, make_llm
                )

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
                        catalog=catalog,
                    )
                )
                registry.start_reaper()
                restored = _restore_history(registry, catalog)
                app.state.registry = registry
                _announce(settings, registry, restored)
                try:
                    yield
                finally:
                    # 后面由 ExitStack 收：registry（会话 → 容器）→ checkpointer → store → pool
                    app.state.registry = None
        finally:
            # **ExitStack 收完之后**才摘日志：会话/池的收尾告警（关会话失败、关池失败）
            # 仍然该进文件，所以顺序与「建的时候反过来」一致 —— 日志是最后进、最后出的。
            # 用 `finally` 而不是顺序落在末尾：`_open_pool`/`_open_event_store` 失败时
            # 会 `raise` 出这一整块（`_fatal` 之后重抛），顺序语句会被跳过、文件句柄
            # 就泄漏在这个进程里了。
            teardown_logging()

    app = FastAPI(title=PROJECT_NAME, version=VERSION, lifespan=lifespan)
    app.include_router(routes.router)
    app.include_router(ws.router)
    if STATIC_DIR.is_dir():
        app.mount("/", _HashedAssetStaticFiles(directory=STATIC_DIR, html=True), name="ui")
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
        logger.info("提示：SWEEP_SANDBOX_ON_START=0 —— 跳过启动清扫，遗留容器不会被回收")
        return

    removed, error = sweep_orphan_containers()
    if error:
        # 清扫失败不挡启动：daemon 没起时后面建会话自然会给出更具体的报错。
        # 原文案是「提示：」但走的 stderr，这里保持**同一个流**用 warning（见
        # `config/logging_setup.py` 的分流规则：info→stdout、warning 及以上→stderr）。
        logger.warning("提示：启动清扫未完成（%s）", error)
    elif removed:
        logger.info("启动清扫：回收了 %d 个遗留沙箱容器", removed)


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
        # 同 `_sweep_orphans`：原文案是「提示：」但走 stderr，用 warning 保持同一个流。
        logger.warning(
            "提示：PERSISTENCE_BACKEND=sqlite —— 事件只存在进程内存里"
            "（GET /sessions/{id}/events 与 WS 回填读的就是它），**服务重启即丢**。\n"
            "      要持久化事件流：docker compose up -d 后设 PERSISTENCE_BACKEND=postgres"
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


def _open_catalog(
    settings: Settings,
    stack: ExitStack,
    pool: Any | None,
    event_store: PostgresEventStore | None,
    checkpointer: Any | None,
    make_llm: Callable[[str], Any] | None,
) -> SessionCatalog:
    """建读路径的探针（P9）。**两个后端都建**——没有它就没有会话目录。

    `checkpointer` 的传法很关键：sqlite 传**共享的那一个实例**（自己再建一个指向同一
    文件会各持一把锁，见模块 docstring），postgres 传 None（catalog 在池上自建一个
    `PostgresSaver(pool)`，实例成本为零，且 `delete_thread` 需要它）。
    """
    return stack.enter_context(
        SessionCatalog(
            settings,
            pool=pool,
            event_store=event_store,
            checkpointer=checkpointer,
            make_llm=make_llm,
            workspace_root=settings.workspace_root,
        )
    )


def _restore_history(registry: SessionRegistry, catalog: SessionCatalog) -> int:
    """把持久化里的历史会话登记进注册表（P9）。返回恢复条数（启动横幅用）。

    **失败只告警，绝不挡启动**：历史会话读不出来，退化成 P7 的「列表从空开始」，
    服务仍然是好的——而那正是用户此刻已有的能力，不该因为一次 DB 抖动就彻底起不来。
    `catalog.discover()` 内部已经把每种失败都降级过了，这里是最后一道兜底。
    """
    try:
        return registry.discover(catalog)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "警告: 恢复历史会话失败，本次列表不含历史会话: %s", e, exc_info=True
        )
        return 0


def _warm_llm() -> None:
    """预热 `langchain_openai` 的 import（实测 ≈8.7s）。

    真正的冷启动成本在 import，不在建图（建图 ~60ms）。放在 lifespan 里跑一次，
    否则第一个会话的第一条指令要莫名多等 8.7 秒。测试注入 make_llm 时跳过。
    """
    started = time.perf_counter()
    import langchain_openai  # noqa: F401  仅为预热 import

    logger.info(
        "预热：langchain_openai 已加载（%.1fs）", time.perf_counter() - started
    )


def _announce(settings: Settings, registry: SessionRegistry, restored: int) -> None:
    """启动横幅（可观测：一眼看清后端、沙箱模式、回收策略、**恢复了几条历史**）。"""
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
    # 恢复历史这一行必须把**能力差异**说透：sqlite 下会话能列出来，但点进去
    # 没有事件流；用户看到的空时间线是他的历史真的没了，不是界面 bug。
    if registry.history_available:
        history = f"已恢复 {restored} 个（事件流可跨重启回放）"
    else:
        history = (
            f"已恢复 {restored} 个（**仅会话，事件不落库**——"
            "sqlite 后端点进去时间线是空的，那是真的没有）"
        )
    # 原来是带 `flush=True` 的 print —— `logging.StreamHandler` 每条记录后自动 flush，
    # 语义不变（而且现在还会同时落进日志文件，这才是一眼看清后端配置的地方）。
    logger.info(
        "%s %s 服务已就绪\n"
        "  持久化：%s\n"
        "  沙箱  ：%s\n"
        "  工作目录：%s（每会话一个子目录）\n"
        "  空闲回收：%.0fs 未活动（running 的会话永不回收）\n"
        "  历史会话：%s\n"
        "  API   ：POST /sessions、POST /sessions/{id}/messages、"
        "POST /sessions/{id}/approval、WS /sessions/{id}/ws\n"
        "  测试页：http://127.0.0.1:8000/",
        PROJECT_NAME,
        VERSION,
        backend,
        sandbox,
        settings.workspace_root,
        registry.idle_timeout,
        history,
    )


def _fatal(what: str, e: Exception) -> None:
    """启动期基础设施失败：给能照着做的提示，然后让 uvicorn 退出（不假装服务是好的）。

    原文案开头的 `\n` 去掉了：那是为了与前面的裸输出隔开一行，而带时间戳的日志
    每条自成一行、已经隔开了；留着只会得到一个挂着格式前缀的空行。
    """
    logger.error("错误：%s。%s", what, e)


def _safe_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


__all__ = ["VERSION", "create_app"]
