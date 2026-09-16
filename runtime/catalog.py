"""P9: 会话目录 —— 从持久化里重建「有哪些会话、各自什么状态」。

P6 让事件与 checkpoint 落了库（`DESIGN.md:304` 的动机之一就是「Web 端要能**回放**一次
会话的完整事件流」），但服务端三个读入口全走进程内存，而 `POST /sessions` 是往注册表里
塞会话的**唯一入口** → 重启即空，库里的数据只有 CLI 的 `--events` 一个消费者。P9 补的
就是这条**读路径**。

## 三件事，各自的位置

1. **目录从哪来**（`discover`）：postgres 走 `events` 表（有 thread_id 也有
   `(thread_id, id)` 索引，一次 GROUP BY 就到手「谁 + 多少条 + 最后活动」）；
   sqlite 后端事件**根本不落库**（关键坑 #46），只能从 checkpoint 反推
   （`list_checkpoint_threads`），代价是恢复出来的是**空壳会话**（`event_count` 为 0）
   ——`api/app.py` 的启动横幅与 `SessionList.history_available` 会明确提示这件事。
2. **状态怎么推**：`SessionStatus` / `_pending_approval` / `_result` / `_error` 全都只在
   内存里，库里没有。所以从 `graph.get_state(config)` 重推（见 `derive_status`）——
   **绝不恢复成 `running`**：`begin()` 只接受 IDLE、`reapable()` 又拒绝 running，
   一个 running 的会话会既不能跑也不能回收，只能 DELETE。
3. **只读探针**：上面那步需要一个**已编译的图**，但**不是**每个会话一个
   （`Session.__init__` 建图 ≈60ms + `WorkspaceManager` 会 mkdir，N 个历史会话就是
   N×60ms + N 个空目录）。这里复用 `build_runtime` 建**一个**探针图，给所有 thread
   读状态；`state` 全在 checkpointer 里，与图对象身份无关（P7 已探针实证）。

## 为什么用 `graph.get_state()` 而不是自己解析裸 checkpoint

`snap.next` / `snap.interrupts` 的推导（versions_seen / pending_sends / interrupt 写在
`checkpoint_writes` 里）是 LangGraph 的内部逻辑，重新实现一遍必然随版本漂移。代价只有
一次性建图 ≈60ms。

## 目录只在这里定义一次

`RestoredRecord.snapshot()` 与 `Session.snapshot()`（`runtime/session.py`）产出**同一个**
字典形状——`api/routes.py`、`api/ws.py` 与 `api/schemas.py` 的 `SessionInfo` 都按它对齐。
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from config.logging_setup import get_logger
from config.settings import Settings
from events.events import Event, EventEmitter
from persistence.checkpointer import build_checkpointer, list_checkpoint_threads
from runtime.assembly import SessionRuntime, build_runtime
from runtime.session import RestorePayload, SessionStatus, summarize_result

logger = get_logger(__name__)

# 探针用的假 thread_id。用不可能与真会话撞的形状（真 id 是 uuid4().hex 或用户自定义）。
PROBE_THREAD_ID = "__catalog_probe__"


class CatalogError(RuntimeError):
    """会话目录的基础设施问题（读不出历史 / 建不出探针图）。"""


@dataclass(frozen=True)
class RestoredRecord:
    """一个持久化里存在、但当前进程**还没有** `Session` 对象的会话。

    刻意是一份**只读摘要**而不是 `Session`：`discover()` 要把库里所有会话都列出来，
    而每个 `Session` 的构造要建图（≈60ms）并 mkdir 一个 workspace 目录。物化推迟到
    `SessionRegistry.get()` —— 真有人点开它的时候。

    `last_id` 只用于**排序**（会话列表里「未物化」那一段按它倒序），不是 seq ——
    seq 是会话内的内存下标，与事件表那个全局 BIGSERIAL 完全是两回事。
    """

    thread_id: str
    status: SessionStatus
    approval: tuple[dict, ...] = ()
    summary: dict | None = None
    event_count: int = 0
    # 未物化时按此倒序（越大越近）。0 = 「不知道」：sqlite 没有 id，被回收的会话
    # 也不查库（它已经静止到超时了，排最后是对的）。
    last_id: int = 0
    # 只在**被回收的会话**上非空（`registry._remember` 把内存里的事件一并带上，
    # 否则 sqlite 后端下「回收」就等于「历史消失」）。`discover()` 一律留空 ——
    # 那时还没有人去读事件流，读了也是白读（N 个会话 × 上百条）。
    events: tuple[Event, ...] = ()

    def snapshot(self) -> dict:
        """与 `Session.snapshot()` **同一个形状**（见模块 docstring）。

        `error` 刻意留 `None`：活的会话里 `_error` 只在「抛异常」时置位，而图 state 的
        `status="error"` 走的是 `result.error`（见 `Session._run`）。同一个会话重启前后
        必须长得一样，所以这里照着**活的那条路径**对齐，而不是自作主张填一个。
        """
        return {
            "thread_id": self.thread_id,
            "status": self.status.value,
            "approval": [dict(p) for p in self.approval],
            "result": self.summary,
            "error": None,
            "event_count": self.event_count,
        }


def derive_status(
    values: dict, *, has_next: bool, has_interrupts: bool
) -> SessionStatus:
    """图 state → 恢复出来的会话状态（纯函数，单测直接断言）。

    | checkpoint 的形状 | 恢复成 | 依据 |
    |---|---|---|
    | 有 interrupt | `AWAITING_APPROVAL` | 停在审批点的会话 checkpoint 是**完整**的，跨进程 `Command(resume=...)` 可以续（决定⑥） |
    | 有未执行节点，或 state 还写着 `running` | `INTERRUPTED` | 进程被杀时正在跑：**恢复成 running 会永久卡死**，所以标成只读的「已中断」（决定②） |
    | 其余（`finished`/`error`/压根没 checkpoint） | `IDLE` | 能看 + 能续跑（决定③） |

    ⚠️ 判序不能反：停在审批点的会话同样满足「有未执行节点」和「state 写着 running」
    （interrupt 就在 tools 节点里抛出），所以 `has_interrupts` 必须**先判**。
    """
    if has_interrupts:
        return SessionStatus.AWAITING_APPROVAL
    if has_next or values.get("status") == "running":
        return SessionStatus.INTERRUPTED
    return SessionStatus.IDLE


def record_from_snapshot(
    thread_id: str, snap: Any, event_count: int, *, last_id: int = 0
) -> RestoredRecord:
    """`StateSnapshot` → `RestoredRecord`（纯函数，单测直接断言）。"""
    values = getattr(snap, "values", None) or {}
    # 与 `driver.run_task` 取 payload 的表达式**逐字一致**（`[(it.value or {}) ...]`），
    # 否则恢复出来的 `approval` 与挂起时推给前端的形状会不一样。
    approval = tuple(
        dict(getattr(it, "value", None) or {})
        for it in (getattr(snap, "interrupts", None) or ())
    )
    return RestoredRecord(
        thread_id=thread_id,
        status=derive_status(
            values,
            has_next=bool(getattr(snap, "next", None)),
            has_interrupts=bool(approval),
        ),
        approval=approval,
        summary=summarize_result(values),
        event_count=event_count,
        last_id=last_id,
    )


def _no_runner(settings: Settings, ws: Any) -> None:
    """探针的 runner 工厂：**返回 None**（= local 空跑）。

    探针只读状态、从不执行工具，所以绝不能给它造一个 `DockerCommandRunner` ——
    它的 `__init__` 会 `atexit.register` 一个绑定方法，多造一个就多泄漏一个对象
    （关键坑 #47）。
    """
    return None


class SessionCatalog:
    """会话目录：枚举 + 状态重建 + 事件回放 + 彻底删除。

    生命周期由调用方持有（`api/app.py` 的 ExitStack），自己不管；`close()` 幂等。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        pool: Any | None = None,
        event_store: Any | None = None,
        checkpointer: Any | None = None,
        make_llm: Callable[[str], Any] | None = None,
        workspace_root: Path | str | None = None,
    ):
        self.settings = settings
        self._pool = pool
        self._event_store = event_store
        self._make_llm = make_llm
        self._workspace_root = Path(workspace_root or settings.workspace_root)

        self._stack: ExitStack | None = ExitStack()
        # **传进来的是调用方的资产**（与 build_runtime 的所有权表一致）：sqlite 后端
        # 必须传全进程共享的那一个 `SqliteSaver`（再建一个指向同一文件会各持一把锁，
        # 关键坑 #46）；postgres 后端调用方给 None，这里在共享池上自建一个
        # `PostgresSaver(pool)`（实例成本为零，且 `delete_thread` 需要它）。
        if checkpointer is None:
            checkpointer = self._stack.enter_context(
                build_checkpointer(settings, pool=pool)
            )
        self._checkpointer = checkpointer
        self._probe: SessionRuntime | None = None

    # ------------------------------------------------------------ 生命周期

    def close(self) -> None:
        stack, self._stack = self._stack, None
        self._probe = None
        self._checkpointer = None
        if stack is not None:
            stack.close()

    def __enter__(self) -> "SessionCatalog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ 能力

    @property
    def history_available(self) -> bool:
        """能不能回放**事件流**（决定④的提示依据）。

        sqlite 后端下事件只在进程内存里，重启即丢 —— 会话能列出来（从 checkpoint 反推），
        但点进去只有空时间线。这个区别必须让用户看见，不能假装一样。
        """
        return self._event_store is not None

    # ------------------------------------------------------------ 目录

    def discover(self) -> list[RestoredRecord]:
        """列出持久化里的全部会话，**按最后活动倒序**。

        **永不抛异常**：启动时读不出历史只该降级并告警，不该让整个服务起不来
        （与 `_sweep_orphans` 同一个取舍）。单个会话的状态重建失败也不丢它 ——
        降级成 `interrupted`（只读），因为「读不出来」时**只读是唯一安全的假设**。
        """
        try:
            threads = self._enumerate()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "警告: 读取历史会话列表失败，本次不恢复历史: %s", e, exc_info=True
            )
            return []
        if not threads:
            return []

        try:
            graph = self._ensure_probe().graph
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "警告: 历史会话恢复所需的图装配失败，本次不恢复历史: %s", e, exc_info=True
            )
            return []

        records: list[RestoredRecord] = []
        for thread_id, event_count, last_id in threads:
            try:
                snap = graph.get_state({"configurable": {"thread_id": thread_id}})
                records.append(
                    record_from_snapshot(
                        thread_id, snap, event_count, last_id=last_id
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "警告: 会话 %s 的状态重建失败，按「已中断」只读处理: %s",
                    thread_id,
                    e,
                    exc_info=True,
                )
                records.append(
                    RestoredRecord(
                        thread_id=thread_id,
                        status=SessionStatus.INTERRUPTED,
                        event_count=event_count,
                        last_id=last_id,
                    )
                )
        return records

    def load_events(self, thread_id: str) -> list[Event]:
        """按写入顺序读回一个会话的完整事件流（sqlite 后端返回空）。

        读回来的顺序**就是 seq 的来源**：灌进 `Session._events` 后下标即 seq，
        与实时追加天然连续（见 `runtime/session.py` 的 `_on_event`）。
        """
        if self._event_store is None:
            return []
        return self._event_store.load(thread_id)

    def restore_payload(self, record: RestoredRecord) -> RestorePayload:
        """`RestoredRecord` + 事件流 → 构造 `Session` 需要的载荷。

        事件在这里一次读全：恢复出来的会话拿到的是**完整历史**，而 `Session` 一侧
        只是把它灌进 `_events`，`event_envelopes()` / WS 回填 / `?since` 全部照旧。

        事件源的优先级：记录里带着的（回收的会话，`registry._remember` 从内存搬的）
        > 从库里读。两者在正常情况下**内容相同**（内存里那份就是写库的那份），
        带在记录里只是为了 sqlite 后端 —— 那里事件从不落库，库读回来永远是空。
        """
        return RestorePayload(
            status=record.status,
            events=record.events or tuple(self.load_events(record.thread_id)),
            summary=record.summary,
            approval=record.approval,
        )

    def purge(self, thread_id: str) -> None:
        """**彻底删除**一个会话的持久化痕迹（checkpoint + 事件）。

        这是 `DELETE /sessions/{id}` 的后半段：目录一旦从库里读，只把会话从内存注册表
        摘掉就会让「删掉的会话重启后复活」。删不掉必须**抛**而不是静默 ——
        返回 204 却什么都没删，等于骗用户。

        **刻意不碰探针图**：删除只依赖 checkpointer，不该因为「探针图装配失败」而删不掉
        （那会把一个纯粹的存储操作绑死在图上）。探针是 `discover`/`get_state` 的需要。
        """
        try:
            self._checkpointer.delete_thread(thread_id)
        except Exception as e:  # noqa: BLE001
            raise CatalogError(f"删除会话 {thread_id} 的 checkpoint 失败: {e}") from e
        if self._event_store is not None:
            try:
                self._event_store.delete_thread(thread_id)
            except Exception as e:  # noqa: BLE001
                raise CatalogError(f"删除会话 {thread_id} 的事件失败: {e}") from e

    # ------------------------------------------------------------ 内部

    def _enumerate(self) -> list[tuple[str, int, int]]:
        """枚举出现过的 thread_id，最近活动在前。返回 `[(thread_id, 事件条数, last_id)]`。"""
        if self._event_store is not None:
            return [
                (t.thread_id, t.event_count, t.last_id)
                for t in self._event_store.list_threads()
            ]
        # sqlite：事件不落库，只能从 checkpoint 反推（`list()` 按 checkpoint_id 倒序，
        # 首次出现顺序即最近活动倒序）。checkpoint 的 id 不是数字，所以 last_id 给 0
        # —— 顺序靠 list 的返回序 + `sorted` 的稳定性保住（见 registry.snapshots）。
        return [(tid, 0, 0) for tid in list_checkpoint_threads(self._checkpointer)]

    def _ensure_probe(self) -> SessionRuntime:
        """懒建只读探针（一次装配，所有 thread 共用）。"""
        if self._probe is not None:
            return self._probe
        if self._stack is None:
            raise CatalogError("会话目录已关闭")
        self._probe = self._stack.enter_context(
            build_runtime(
                self.settings,
                thread_id=PROBE_THREAD_ID,
                workspace_root=self._workspace_root,
                make_llm=self._make_llm,
                runner_factory=_no_runner,
                checkpointer=self._checkpointer,
                pool=self._pool,
                event_store=self._event_store,
                # 独立 emitter：探针不发事件，但万一发了也不该落进任何会话/默认单例
                emitter=EventEmitter(),
            )
        )
        return self._probe


__all__ = [
    "PROBE_THREAD_ID",
    "CatalogError",
    "RestoredRecord",
    "SessionCatalog",
    "derive_status",
    "record_from_snapshot",
]
