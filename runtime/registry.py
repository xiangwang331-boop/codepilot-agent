"""P7: 会话注册表 —— 服务端的会话目录 + 空闲回收。

CLI 里会话的生命周期是一个 `with` 块（进程退出即结束）；服务端没有这个天然边界，
所以需要有人**持有**会话并按策略释放：容器、checkpointer、event store 监听器都是
每会话一份的资源，没人放手就是泄漏。

## 两件事分开

- `close(id)` —— 显式销毁（DELETE /sessions/{id}）。**关闭、从目录里移除、并删掉
  持久化痕迹**（P9 起「移除」必须是真删，见下）。
- `reap_idle()` —— 空闲回收：状态是 `idle`/`awaiting_approval`/`interrupted` 且静置
  超时的会话一并关掉。**`running` 永不回收**（见 `Session.reapable`）。

## P9：目录有两层，「回收」不再等于「消失」

P7 的注册表就是 `_sessions` 一本账，回收即 404。可持久化接上之后这个语义崩了：
库里明明有，重启即空（读路径缺失），回收即 404（历史消失）。所以现在是两层：

- `_sessions`：**已物化**的会话（有 graph、有容器、能跑）。这一层是缓存。
- `_known`：**未物化记录**（`catalog.RestoredRecord`），即「持久化里有、本进程还没
  把它建起来」的会话。改它需要同时改库，所以只有 `discover`/`close`/回收三处会写。

`get(id)` 未命中 `_sessions` 时**在锁内懒物化**一条记录。为什么懒：`Session.__init__`
无条件 `build_runtime`（建图 ≈60ms）并 mkdir 一个 workspace 目录，N 个历史会话就是
N×60ms + N 个空目录；而绝大多数历史会话只是要在列表里有一行。

**回收 ≠ 历史消失**：`reap_idle` 把内存会话摘掉之后会用它的收尾状态回填一条 `_known`
记录（从内存造，**不查库**），否则被回收的会话会从列表里凭空消失——而那正是用户
最初报的那个问题，只是换了个触发点。

## 并发

`create` 与懒物化都在锁内完成（会话构造 ≈60ms，且都是低频动作），换来的是
「检查 → 登记」的原子性：物化若在锁外做，两个并发请求会各建一个 `Session`，
后写的那个赢、先建的那个**永远不会被 close**（泄漏一个 graph）。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from config.logging_setup import get_logger
from config.settings import Settings
from runtime.catalog import RestoredRecord
from runtime.session import Session, SessionStatus

logger = get_logger(__name__)

# 空闲多久回收。默认 30 分钟——远大于一次 LLM 往返，又不会让容器长住不散。
DEFAULT_IDLE_TIMEOUT = 1800.0


class SessionError(RuntimeError):
    """注册表层面的会话错误（如 id 冲突）。"""


class SessionExistsError(SessionError):
    """要创建的会话 id 已被占用。"""


class SessionRegistry:
    def __init__(
        self,
        settings: Settings,
        *,
        make_llm: Callable[[str], Any] | None = None,
        runner_factory: Callable[..., Any] | None = None,
        pool: Any | None = None,
        event_store: Any | None = None,
        checkpointer: Any | None = None,
        require_approval_for: tuple[str, ...] = ("coder",),
        workspace_root: Path | str | None = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
        catalog: Any | None = None,
    ):
        self.settings = settings
        self.idle_timeout = idle_timeout
        self.clock = clock
        # 每个会话一个独立 workspace 目录：`<workspace_root>/<thread_id>`。
        # 这是「每用户对话一个隔离 workspace」的落点（P5 + P7 的共同设计意图）——
        # agent 写的文件、容器 bind mount 的都是这个目录，会话之间不互相看见。
        self.workspace_root = Path(workspace_root or settings.workspace_root)
        self._make_llm = make_llm
        self._runner_factory = runner_factory
        self._pool = pool
        self._event_store = event_store
        # sqlite 后端下这是**全进程共用的那一个** SqliteSaver（多实例指向同一文件才危险），
        # postgres 后端下留 None → 每个会话在共享池上建自己的 PostgresSaver。
        self._checkpointer = checkpointer
        self._require_approval_for = require_approval_for
        # P9：会话目录（枚举历史 + 读事件 + 删持久化）。None = 纯内存模式
        # （所有既有单测都不传它，行为与 P7 逐字一致）。
        self._catalog = catalog

        self._sessions: dict[str, Session] = {}
        # P9：持久化里有、本进程还没物化的会话（见模块 docstring）。**回收线程不碰它。**
        self._known: dict[str, RestoredRecord] = {}
        self._lock = threading.Lock()
        self._reaper: threading.Thread | None = None
        self._stop_reaper = threading.Event()

    # ------------------------------------------------------------ 能力

    @property
    def history_available(self) -> bool:
        """重启后还能不能读回**事件流**（决定④的提示依据，路由只读这一个口）。

        没有 catalog（纯内存注册表：全部既有单测）也是 False —— 那种进程重启后
        连会话本身都没有，谈不上「历史可用」。sqlite 后端同样是 False：会话能从
        checkpoint 反推出来（空壳），但事件从不落库（关键坑 #46）。
        """
        return self._catalog is not None and bool(self._catalog.history_available)

    # ------------------------------------------------------------ 目录

    def discover(self, catalog: Any | None = None) -> int:
        """把持久化里的历史会话登记成未物化记录（P9）。返回**新登记**的条数。

        启动时调一次。**绝不抛异常**（`catalog.discover()` 自己已经保证降级）：
        读不出历史只该在列表里少几行，不该让整个服务起不来。

        已在 `_sessions`/`_known` 里的跳过 —— 本进程刚建的会话比库里的新，
        拿库里的记录去覆盖它等于把新的状态说成旧的。
        """
        if catalog is not None:
            self._catalog = catalog
        if self._catalog is None:
            return 0
        records = self._catalog.discover()
        with self._lock:
            added = 0
            for record in records:
                if record.thread_id in self._sessions or record.thread_id in self._known:
                    continue
                self._known[record.thread_id] = record
                added += 1
        if added:
            logger.info(
                "恢复历史会话：登记 %d 个未物化记录（库里共 %d 条）",
                added,
                len(records),
            )
        return added

    def create(self, thread_id: str | None = None) -> Session:
        """新建一个会话。id 冲突抛 `SessionExistsError`。

        P9：冲突判定**同时看 `_known`**。只查 `_sessions` 的话，`POST /sessions
        {"thread_id": "<库里已有的 id>"}` 会成功并造出一个**没有历史**的新会话压在
        同一条 thread 上 —— 图形 state 是真的（`existing_message_count` 从库里读到
        旧历史，续跑会接上去），但新会话的 `_events` 是空的，**用户看不到刚续上的
        那段历史**。宁可 409，让他走 `POST /messages` 续跑。
        """
        thread_id = thread_id or uuid4().hex
        with self._lock:
            if thread_id in self._sessions or thread_id in self._known:
                raise SessionExistsError(f"会话 {thread_id} 已存在")
            session = Session(
                thread_id,
                self.settings,
                workspace_root=self.workspace_root / thread_id,
                make_llm=self._make_llm,
                runner_factory=self._runner_factory,
                require_approval_for=self._require_approval_for,
                pool=self._pool,
                event_store=self._event_store,
                checkpointer=self._checkpointer,
                clock=self.clock,
            )
            self._sessions[thread_id] = session
        logger.info(
            "建会话 %s（沙箱=%s，工作目录=%s）",
            thread_id,
            self.settings.sandbox_mode,
            session.workspace_root,
        )
        return session

    def get(self, thread_id: str) -> Session | None:
        """取会话，**未命中就懒物化**历史记录（P9）。返回 None = 路由的 404。

        这一层是「重启后还能点开历史会话」的全部机制：路由 / WS handler / 审批
        端点全都只调 `get()`，它们一行都不用改（`ws.py` 的 4404「不会再有事件」
        判定也因此自动变正确 —— 历史会话是有订阅价值的）。
        """
        with self._lock:
            session = self._sessions.get(thread_id)
        if session is not None:
            return session
        return self._materialize(thread_id)

    def _materialize(self, thread_id: str) -> Session | None:
        """把一条未物化记录变成真会话（P9）。

        **全程持锁**（与 `create` 同一个权衡）：锁外做的话两个并发请求会各建一个
        `Session`，后写的赢、先建的那个永远不会被 close —— 泄漏一个 graph。
        锁内做也就多一次事件流的库读取。

        失败**返回 None（→ 404）而不是抛**：一条坏记录不该让 `GET /sessions/{id}`
        返回 500；记录放回去，下次访问再试（故障可能是瞬时的，如 sqlite 锁占用）。
        """
        catalog = self._catalog
        with self._lock:
            if thread_id in self._sessions:  # 等锁期间被别人物化了
                return self._sessions[thread_id]
            record = self._known.pop(thread_id, None)
            if record is None or catalog is None:
                return None
            try:
                session = Session(
                    thread_id,
                    self.settings,
                    workspace_root=self.workspace_root / thread_id,
                    make_llm=self._make_llm,
                    runner_factory=self._runner_factory,
                    require_approval_for=self._require_approval_for,
                    pool=self._pool,
                    event_store=self._event_store,
                    checkpointer=self._checkpointer,
                    clock=self.clock,
                    restore=catalog.restore_payload(record),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "警告: 历史会话 %s 恢复失败，本次按不存在处理: %s", thread_id, e,
                    exc_info=True,
                )
                self._known[thread_id] = record
                return None
            self._sessions[thread_id] = session
            # 用 record.event_count 而不是 len(session.events)：后者要拿会话锁，
            # 而这里正**持着注册表锁** —— 不制造「注册表锁 → 会话锁」的嵌套。
            logger.info(
                "物化历史会话 %s（状态=%s，%d 条事件）",
                thread_id,
                record.status.value,
                record.event_count,
            )
            return session

    def sessions(self) -> list[Session]:
        """**只有已物化的**（内部用：回收、关停、计数）。全量目录看 `snapshots()`。"""
        with self._lock:
            return list(self._sessions.values())

    def snapshots(self) -> list[dict]:
        """全部会话的状态一览：已物化的在前，未物化的历史记录在后（P9）。

        **分区排序而不是混排**：两段的时钟不可比 —— 内存那段是进程内活动时刻
        （`time.monotonic`），记录那段是事件表的全局 BIGSERIAL（跨进程、跨重启）。
        硬凑成一个序只会得到「看起来随机」的列表。段内各自「最近活动在前」，
        用户真正关心的是「我刚跑的会话在最上面」，这一点分区也能满足。
        """
        with self._lock:
            sessions = list(self._sessions.values())
            records = [
                r for tid, r in self._known.items() if tid not in self._sessions
            ]
        live = sorted(sessions, key=lambda s: s.last_activity, reverse=True)
        pending = sorted(records, key=lambda r: r.last_id, reverse=True)
        return [s.snapshot() for s in live] + [r.snapshot() for r in pending]

    def __len__(self) -> int:
        """**只数已物化的会话**（与 P7 逐字一致）。

        刻意不含 `_known`：这两个容器协议的语义是「注册表还握着没有」——
        `test_session.py` 的 `"idle" not in reg`、`len(reg) == 0` 全靠它，
        而「目录里有哪些」由 `snapshots()` 负责（唯一的全量视图）。
        """
        with self._lock:
            return len(self._sessions)

    def __contains__(self, thread_id: object) -> bool:
        with self._lock:
            return thread_id in self._sessions

    # ------------------------------------------------------------ 释放

    def close(self, thread_id: str) -> bool:
        """销毁一个会话：关资源 + 从目录移除 + **删掉持久化痕迹**。不存在返回 False。

        P9 起「移除」必须是真删（用户决策⑤）：目录改成从库里读之后，只 pop 内存会
        让**删掉的会话重启后复活**——比不能删更糟。所以这里连着 checkpoint 与事件
        一起删。

        删库失败**让它抛**（`CatalogError` → 500）：内存已经摘干净了，但返回 204
        而库里还在，等于告诉用户「删了」而下次重启它又回来。
        """
        with self._lock:
            session = self._sessions.pop(thread_id, None)
            record = self._known.pop(thread_id, None)
        if session is None and record is None:
            return False
        if session is not None:
            session.close()
        if self._catalog is not None:
            self._catalog.purge(thread_id)
        logger.info(
            "删除会话 %s（连库一起删：%s）",
            thread_id,
            "已物化" if session is not None else "仅记录",
        )
        return True

    def reap_idle(self) -> list[str]:
        """回收空闲会话，返回被回收的 id 列表。

        **`running` 的会话不在候选里**——会话可能正卡在一次几十秒的 LLM 调用或
        `docker exec` 上，此时 `rm -f` 容器会把正在跑的命令从底下抽走。

        「判可回收」与「摘出目录」在同一把锁内完成：中间那一瞬正是 worker 可能
        被唤醒去跑东西的窗口，先摘出来才能保证回收不会打到刚跑起来的会话。

        P9：摘掉之后**用它的收尾状态回填一条未物化记录**（`_remember`）—— 回收的
        意思是「放手」，不是「历史消失」。回填必须在 `session.close()` **之前**做：
        close 会把状态置成 `closed` 并清掉 `_rt`。
        """
        reaped: list[str] = []
        for thread_id in [s.thread_id for s in self.sessions()]:
            with self._lock:
                session = self._sessions.get(thread_id)
                if session is None or not session.reapable(self.idle_timeout):
                    continue
                self._sessions.pop(thread_id, None)
            self._remember(session)
            session.close()
            reaped.append(thread_id)
        if reaped:
            logger.info(
                "空闲回收：关掉 %d 个会话（静置超 %.0fs）: %s",
                len(reaped),
                self.idle_timeout,
                ", ".join(reaped),
            )
        return reaped

    def _remember(self, session: Session) -> None:
        """把刚回收的会话记成一条未物化记录（P9）。**从内存造，不查库。**

        事件一并带进记录：sqlite 后端下事件从不落库，不带的话「回收」就等于
        「这个会话的历史永远消失」（`catalog.load_events` 读回来永远是空）。

        `last_id` 留 0 = 「不知道」——被回收意味着它已经静置到超时（默认 30 分钟），
        在未物化那一段里排最后是符合事实的。

        **没有 catalog 时直接不记**：没有持久化目录就没有「重启后还能打开」这回事，
        此时回填只会让 `snapshots()` 凭空多出一行、并让 `__contains__`/`__len__`
        的既有语义（注册表还握着没有）失真。纯内存模式保持 P7 行为。
        """
        if self._catalog is None or session.status is SessionStatus.CLOSED:
            return  # 显式关掉的会话不该复活
        record = RestoredRecord(
            thread_id=session.thread_id,
            status=session.status,
            approval=tuple(session.pending_approval),
            summary=session.snapshot()["result"],
            event_count=len(session.events),
            events=tuple(session.events),
        )
        with self._lock:
            if session.thread_id not in self._sessions:
                self._known[session.thread_id] = record

    def close_all(self) -> None:
        """关停全部会话（服务退出时用）。单个会话关闭失败不影响其余。

        P9 起失败**记日志**（原来纯 `pass`）：关不掉的会话意味着它的沙箱容器与
        文件句柄没人放，而这是进程退出前的最后一道手 —— 静默失败在这里最贵。
        """
        for session in self.sessions():
            try:
                session.close()
            except Exception:  # noqa: BLE001  关停要尽力而为，不能因为一个炸掉而漏掉其余
                logger.warning(
                    "警告: 关停会话 %s 失败，继续关其余会话", session.thread_id,
                    exc_info=True,
                )
        with self._lock:
            self._sessions.clear()

    # ------------------------------------------------------------ 后台回收

    def start_reaper(self, interval: float | None = None) -> None:
        """起后台回收线程（幂等）。默认间隔取 `idle_timeout / 4`，夹在 1~60 秒。

        测试不要用它——直接调 `reap_idle()` 配合注入的假时钟，才是确定性的。
        """
        if self._reaper is not None and self._reaper.is_alive():
            return
        interval = interval or max(1.0, min(60.0, self.idle_timeout / 4))
        self._stop_reaper.clear()

        def loop() -> None:
            while not self._stop_reaper.wait(interval):
                try:
                    self.reap_idle()
                except Exception:  # noqa: BLE001  后台线程绝不能因为一次失败退出
                    # P9 起记日志（原来纯 `pass`）：reaper 是后台线程，它反复失败
                    # 时界面上完全看不出来（容器只增不减），**只有日志能暴露**。
                    logger.warning(
                        "警告: 空闲回收这一轮失败，等下一个周期再试", exc_info=True
                    )

        self._reaper = threading.Thread(
            target=loop, name="codepilot-reaper", daemon=True
        )
        self._reaper.start()

    def stop_reaper(self) -> None:
        self._stop_reaper.set()
        reaper, self._reaper = self._reaper, None
        if reaper is not None:
            reaper.join(timeout=2.0)

    def __enter__(self) -> "SessionRegistry":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop_reaper()
        self.close_all()


__all__ = [
    "DEFAULT_IDLE_TIMEOUT",
    "SessionError",
    "SessionExistsError",
    "SessionRegistry",
]
