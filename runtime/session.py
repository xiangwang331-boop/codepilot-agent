"""P7: 会话 —— 服务端的执行单元（状态机 + worker 线程 + 审批槽 + 订阅者）。

CLI 把「跑一个任务」当成一次函数调用；服务端不能——HTTP 请求必须立刻返回，
而 `graph.invoke()` 全程同步阻塞（LLM、`docker exec` 都是同步的），放 event loop
会卡死整个服务。所以每个会话一个**专用 worker 线程**，事件与状态变化经订阅者
推给 WS。

## 状态机（唯一权威）

```
        begin()                _park_for_approval()          resume()
IDLE ──────────► RUNNING ──────────────────────────► AWAITING_APPROVAL ──► RUNNING
  ▲                 │                                                      │
  └─────────────────┴──────────────── worker 跑完 / 抛异常 ─────────────────┘
                                              close() ↓
                                            CLOSED（终态）

INTERRUPTED（P9）是**另一个来源**，不进上面的循环：它只在「从持久化恢复」时出现
（`restore=`），代表进程上次被杀时这个会话正在跑。**它既不能 begin() 也不能 resume()**
（`begin()` 只认 IDLE）——这是刻意的只读态，界面上标「已中断」给用户看历史。

**`begin()`/`resume()` 一律以状态字段为准，绝不能靠「worker 线程是否还活着」判断忙闲。**
实测（探针实证）：会话挂起在 interrupt 时用普通 input 再 invoke 一次，**不报错**，
`next` 从 `('tools',)` 清成 `()`、`interrupts` 从 1 清成 0，挂起的委派被静默丢弃、
文件从未创建——全程无异常无日志。所以挡并发指令只能靠 `running`/`awaiting_approval`
这两个显式状态（两者都让 `begin()` 返回 False = API 的 409）；线程存活与否根本不可观测。

## 审批「让出线程」而不是空等

worker 撞到 interrupt 时 `on_interrupt` 返回 `None` → 记录 payload、置
`AWAITING_APPROVAL`、通知订阅者、**worker 线程结束**（不占线程）。收到审批后
`resume()` 起一个**新**线程，用**同一个 graph 对象** `graph.invoke(Command(resume=...))`
继续——state 全在 checkpointer 里，与 graph 对象身份无关（已探针实证）。

⚠️ resume 值必须经 `driver.normalize_answer` 归一化：`supervisor.py` 的判定是
`if answer != "yes"`，`Command(resume=True)` 会**静默**变成「用户拒绝」。

## 订阅协议（WS 信封）

订阅者收到的是**信封字典**，不是裸 `Event`——这样 `api/ws.py` 只管 `json.dumps`，
协议形状在这一层就可测（不需要起 HTTP）：

```json
{"kind": "status", "thread_id": "...", "status": "awaiting_approval", "approval": [...]}
{"kind": "event",  "seq": 12, "event": {"type": "ToolCallStarted", "agent": "coder", ...}}
{"kind": "error",  "message": "..."}
```

推的是**结构化 Event**而不是 `format_event()` 的渲染结果：后者有损（AGENT_STARTED /
AGENT_COMPLETED 丢 message、tool 事件丢 `[agent]` 前缀归属、CONDENSE 变多行字符串），
而 UI 要的恰恰是「哪个 agent 在干什么」。

## 线程绑定必须在每个 worker 启动时重做

`bind_thread` / `bind_emitter` 是**线程上下文作用域**，不是会话作用域——线程不继承
别的线程的绑定，所以两处绑定收敛在 `_start_worker` 一处，审批恢复新起的线程同样走它。
"""
from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command

from config.logging_setup import get_logger
from config.settings import Settings
from events.events import Event, EventEmitter, EventType, bind_emitter, bind_thread
from runtime.assembly import SessionRuntime, build_runtime
from runtime.driver import normalize_answer, run_task

logger = get_logger(__name__)

# 订阅者签名：收一个 WS 信封字典（见模块 docstring）。
Subscriber = Callable[[dict], None]

#: 日志里任务正文的截断长度。**日志是「动作级」的**：需求全文由事件流承载
#: （`events` 表 + WS），这里只要够认出「是哪一次下发」。
_TASK_LOG_CHARS = 80


def _ellipsis(text: str, limit: int = _TASK_LOG_CHARS) -> str:
    """把长文本截成一行日志能装下的样子（单行化 + 省略号）。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    CLOSED = "closed"
    # P9：从持久化恢复出来的、上次没跑完就随进程死掉的会话。**只读**（见模块 docstring）。
    INTERRUPTED = "interrupted"


def jsonable(value: Any) -> Any:
    """把任意值转成能过裸 `json.dumps` 的形状。

    WS 那条路是 **Starlette 的 `send_json`**（裸 `json.dumps`，没有 FastAPI 的
    `jsonable_encoder` 兜底），任何非 JSON 原生类型都会在那里抛 —— 而抛点在下游，
    异常被订阅路径的 try 吞掉，表现是**客户端一个字都收不到、服务端毫无日志**。
    所以进信封的东西必须先过这一层。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def event_payload(e: Event) -> dict:
    """把事件转成推给前端的结构化字典（见模块 docstring：不用 format_event）。"""
    d = asdict(e)
    d["type"] = getattr(e.type, "value", e.type)
    d["detail"] = jsonable(d["detail"])
    return d


def summarize_result(values: Any) -> dict | None:
    """图 state → 紧凑摘要。**`_result_summary_locked()` 与 P9 的恢复路径共用这一个定义。**

    刻意不是整包 state，两个理由都是真机踩出来的：

    1. state 里的 `messages` 是 LangChain 消息对象。WS 走的是 Starlette 的裸
       `json.dumps` → `TypeError` → 异常被订阅路径吞掉 → **socket 静默死掉**
       （回填一个跑过任务的会话，客户端一个字都收不到，服务端没日志）；
    2. status 信封**每次状态变化都推一次**，塞整段历史会随会话无限膨胀
       （messages 里还有 2KB 的 supervisor system prompt）。

    完整历史看事件流：`GET /sessions/{id}/events` 或 WS 回填。

    恢复路径（`runtime/catalog.py`）也必须过这里 —— 否则同一个会话在重启前后会给出
    两个形状不同的 `result`，前端就得写两套解析。
    """
    if not isinstance(values, dict):
        return None
    messages = values.get("messages") or []
    return {
        "status": values.get("status"),
        "result": jsonable(values.get("result")),
        "error": jsonable(values.get("error")),
        "iteration_count": values.get("iteration_count"),
        "message_count": len(messages),
    }


@dataclass(frozen=True)
class RestorePayload:
    """「带历史出生」要写进 `Session` 的全部东西（P9）。

    `Session.__init__` 在 `build_runtime` **成功之后**才写它 —— 时机很关键：
    装配期间不发任何事件，所以灌历史不会触发 `event_store.record` 二次落库，
    也不会惊动订阅者（此刻还没有订阅者）。

    `events` 的**顺序就是 seq 的来源**：按序灌进 `_events` 之后下标天然是 `0..N-1`，
    新事件接着 `N` 往下 —— `event_envelopes()` / WS 回填 / `?since` 闭区间游标
    全部不用改（见 `runtime/catalog.py` 的模块 docstring）。
    """

    status: SessionStatus
    events: tuple[Event, ...] = ()
    summary: dict | None = None
    approval: tuple[dict, ...] = ()


class Session:
    """一个会话的全部运行时状态。构造即装配（fail fast），`close()` 释放。"""

    def __init__(
        self,
        thread_id: str,
        settings: Settings,
        *,
        workspace_root: Path | str,
        make_llm: Callable[[str], Any] | None = None,
        runner_factory: Callable[..., Any] | None = None,
        require_approval_for: tuple[str, ...] = ("coder",),
        pool: Any | None = None,
        event_store: Any | None = None,
        checkpointer: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        restore: RestorePayload | None = None,
    ):
        self.thread_id = thread_id
        self.settings = settings
        self.workspace_root = Path(workspace_root)
        self.clock = clock

        # 会话专属 emitter：worker 线程绑定它，图内所有 emit 自动落到这里，
        # 与 CLI 的模块级单例、与别的会话完全隔开。
        self.emitter = EventEmitter()

        self._lock = threading.RLock()
        self._status = SessionStatus.IDLE
        # 订阅者连同「回填到达的序号」一起存：推送是锁外做的，所以一条事件可能
        # 先被追加进 _events（回填能看见它）再推给订阅者 —— start 就是用来把
        # 这批重复推给掐掉的（见 subscribe / _on_event）。
        self._subscribers: list[tuple[Subscriber, int]] = []
        # 已经报过「推送失败」的订阅者（按 id(fn)）。**只为把日志限成一次**：
        # 坏掉的订阅者会对之后每一条事件都抛，逐条记会把日志刷成同一行 ——
        # 首条带栈就够定位了，见 `_publish`。
        self._dead_subscribers: set[int] = set()
        self._events: list[Event] = []
        self._pending_approval: list[dict] = []
        self._worker: threading.Thread | None = None
        self._result: dict | None = None
        self._error: str | None = None
        self._sandbox_entered = False
        self._last_activity = clock()
        # P9：恢复出来的会话没有「刚跑完的 result」，摘要在建会话时就定好了。
        # 一旦本进程真跑完一轮，_run 会把它清掉、改用 _result（新结果覆盖旧摘要）。
        self._restored_summary: dict | None = None

        # 会话自己持有整段生命周期：装配 + 沙箱容器都挂在同一个 ExitStack 上。
        # 沙箱**跨 worker** 存活（审批让出线程时不能把容器 rm -f），所以它不跟着
        # worker 的 with 走，而是跟着会话走（见 assembly 的所有权表 —— runner
        # 传进来时是调用方的资产，这里只负责进/出 rt.sandbox()）。
        self._stack: ExitStack | None = ExitStack()
        self._rt: SessionRuntime | None = None
        self.emitter.add_listener(self._on_event)
        try:
            self._rt = self._stack.enter_context(
                build_runtime(
                    settings,
                    thread_id=thread_id,
                    workspace_root=self.workspace_root,
                    make_llm=make_llm,
                    runner_factory=runner_factory,
                    require_approval_for=require_approval_for,
                    pool=pool,
                    event_store=event_store,
                    checkpointer=checkpointer,
                    emitter=self.emitter,
                )
            )
        except BaseException:
            self._stack.close()  # 构造失败也要退出已进入的上下文
            self._stack = None
            raise

        if restore is not None:
            self._apply_restore(restore)

    def _apply_restore(self, payload: RestorePayload) -> None:
        """把持久化的历史写进这个会话（P9，构造末尾调用一次）。

        历史**必须先于任何订阅者**灌进来：seq 是 `_events` 的下标推导值
        （`_on_event` 的 `len(self._events) - 1`、`_event_envelopes_locked` 的
        `enumerate`、`subscribe` 的水位线 `len(self._events)`），所以只要历史此刻
        已经在 `_events` 里，它天然就是 `0..N-1`、后续新事件接着 `N` 往下——
        路由、WS 回填、`?since` 游标一行都不用改。

        此刻确实还没有订阅者：`subscribe()` 是路由/WS handler 才会调的，而对象刚构造完。
        """
        with self._lock:
            self._events.extend(payload.events)
            self._status = payload.status
            self._restored_summary = payload.summary
            self._pending_approval = [dict(p) for p in payload.approval]

    # ------------------------------------------------------------ 只读视图

    @property
    def status(self) -> SessionStatus:
        with self._lock:
            return self._status

    @property
    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    @property
    def pending_approval(self) -> list[dict]:
        with self._lock:
            return [dict(p) for p in self._pending_approval]

    @property
    def runner(self) -> Any | None:
        return self._rt.runner if self._rt else None

    @property
    def subscriber_count(self) -> int:
        """当前订阅者数（可观测 + 测试断言「WS 断开后没留下悬挂订阅」）。"""
        with self._lock:
            return len(self._subscribers)

    def snapshot(self) -> dict:
        """状态一览（WS 的 status 信封、REST 的 GET /sessions/{id} 共用）。"""
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "status": self._status.value,
            "approval": [dict(p) for p in self._pending_approval],
            "result": self._result_summary_locked(),
            "error": self._error,
            "event_count": len(self._events),
        }

    def _result_summary_locked(self) -> dict | None:
        """本会话的结果摘要。形状定义在 `summarize_result`（实时与恢复共用）。"""
        if self._restored_summary is not None:
            return self._restored_summary
        return summarize_result(self._result)

    # ------------------------------------------------------------ 订阅

    def subscribe(self, fn: Subscriber) -> tuple[dict, list[dict]]:
        """注册订阅者，返回 `(当前状态信封, 到目前为止的全部事件信封)`。

        回填与「注册点」在同一把锁下取，因此**既不会漏也不会重**：`start` 之后的
        事件才推给它，`start` 之前的已经全在回填里了。
        """
        with self._lock:
            status_env = {"kind": "status", **self._snapshot_locked()}
            backlog = self._event_envelopes_locked()
            self._subscribers.append((fn, len(self._events)))
            return status_env, backlog

    def event_envelopes(self) -> list[dict]:
        """到目前为止的全部事件信封（与 WS 回填**同一形状**，`GET /events` 也用它）。

        形状只在这一个地方定义：REST 与 WS 各建一套的话，前端得写两套解析。
        """
        with self._lock:
            return self._event_envelopes_locked()

    def _event_envelopes_locked(self) -> list[dict]:
        return [self._event_env(i, e) for i, e in enumerate(self._events)]

    def unsubscribe(self, fn: Subscriber) -> None:
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s[0] is not fn]
            self._dead_subscribers.discard(id(fn))

    @staticmethod
    def _event_env(seq: int, e: Event) -> dict:
        return {"kind": "event", "seq": seq, "event": event_payload(e)}

    def _publish(self, envelope: dict, *, seq: int | None = None) -> None:
        """推一个信封给所有订阅者。**推送失败绝不反噬会话。**

        `EventEmitter.emit` 是裸的 `for fn in listeners: fn(e)`，监听器抛异常会直接
        冒泡进图节点炸掉正在跑的 agent（P6 已踩过同类坑）——WS 的对端随时可能断开。

        ⚠️ 推送在**锁外**做，所以「追加 `_events`」与「推给订阅者」之间有窗口：
        刚好在这个窗口里注册的订阅者，会同时从回填和实时推送拿到同一条事件。
        用 `seq < start` 掐掉 —— 这条判据在锁内取值，两种交错都只有一个来源。
        """
        with self._lock:
            subscribers = list(self._subscribers)
        for fn, start in subscribers:
            if seq is not None and seq < start:
                continue
            try:
                fn(envelope)
            except Exception:  # noqa: BLE001
                self._note_dead_subscriber(fn)

    def _note_dead_subscriber(self, fn: Subscriber) -> None:
        """订阅者推送失败：**每个订阅者只记一次**（带栈），之后不再重复报。

        WS 对端随时可能断开（`loop.call_soon_threadsafe` 打在已关闭的 loop 上会抛
        `RuntimeError`），一个坏订阅者会对**之后每一条**事件都失败 —— 逐条记会让
        日志变成同一行刷屏，而这一行本身没有任何新信息。首条带栈就够定位了。
        """
        key = id(fn)
        with self._lock:
            if key in self._dead_subscribers:
                return
            self._dead_subscribers.add(key)
        logger.warning(
            "警告: 会话 %s 的一个事件订阅者推送失败（后续失败不再重复报告）",
            self.thread_id,
            exc_info=True,
        )

    def _publish_status(self) -> None:
        """状态信封是快照、天然幂等，没有判重问题，所有订阅者都收。"""
        try:
            self._publish({"kind": "status", **self.snapshot()})
        except Exception:  # noqa: BLE001
            logger.warning(
                "警告: 会话 %s 推送状态信封失败（状态机不受影响）",
                self.thread_id,
                exc_info=True,
            )

    def _on_event(self, e: Event) -> None:
        with self._lock:
            self._events.append(e)
            seq = len(self._events) - 1
        self._publish(self._event_env(seq, e), seq=seq)

    # ------------------------------------------------------------ 执行

    def begin(self, task: str) -> bool:
        """开始一个任务。会话忙（running/awaiting_approval/closed）时返回 False。

        False 就是 API 的 409 —— 这里绝不能「先试试看」：挂起中再 invoke 一次，
        LangGraph 会**静默吞掉**挂起的 interrupt（见模块 docstring）。
        """
        with self._lock:
            if self._status is not SessionStatus.IDLE:
                return False
            self._status = SessionStatus.RUNNING
            self._touch()

        try:
            self._ensure_sandbox()
            graph_input, announce = self._graph_input(task)
        except BaseException as e:
            with self._lock:
                self._status = SessionStatus.IDLE
                self._error = f"{type(e).__name__}: {e}"
            # 起沙箱失败（daemon 没开 / 镜像缺）与「构造图输入失败」都走这里，
            # 对用户是一句 HTTP 错误，对我们只有这个栈能说明原因。
            logger.error(
                "会话 %s 无法开始执行: %s: %s", self.thread_id, type(e).__name__, e,
                exc_info=True,
            )
            raise

        self._publish_status()
        self._start_worker(graph_input, announce=announce)
        # 任务正文只截前 80 字：日志是「动作级」的，需求全文由事件流承载
        logger.info("下发指令 %s: %s", self.thread_id, _ellipsis(task))
        return True

    def resume(self, answer: Any = True) -> bool:
        """批准/拒绝一个挂起的委派。不在 `AWAITING_APPROVAL` 时返回 False。

        `answer` 可以是 bool（API 的 `{"approved": true}`）或字符串，统一经
        `normalize_answer` 归一化成精确的 `"yes"`/`"no"`（见模块 docstring 的坑）。
        """
        with self._lock:
            if self._status is not SessionStatus.AWAITING_APPROVAL:
                return False
            self._status = SessionStatus.RUNNING
            self._pending_approval = []
            self._touch()

        resolved = normalize_answer(answer)
        self._publish_status()
        self._start_worker(Command(resume=resolved), announce=False)
        # 记归一化**之后**的值：`Command(resume=True)` 会被静默当成拒绝（关键坑 #32），
        # 日志里看见的必须是真的送进图的那个字符串，而不是调用方传来的形状。
        logger.info("审批 %s: %s", self.thread_id, resolved)
        return True

    def close(self) -> None:
        """释放会话资源（关容器、关 checkpointer）。幂等。"""
        with self._lock:
            if self._status is SessionStatus.CLOSED:
                return
            self._status = SessionStatus.CLOSED
            stack, self._stack = self._stack, None
            rt, self._rt = self._rt, None
        if stack is not None:
            try:
                stack.close()
            finally:
                self._publish_status()
        if rt is not None and rt.runner is not None:
            # 沙箱没进过（会话从没跑过东西）时 `stack.close()` 不会碰 runner，
            # 但 atexit 注册的绑定方法是**构造时**挂上的强引用 —— 必须在这里摘掉，
            # 否则每建一个会话就永久泄漏一个 runner 对象。stop() 幂等。
            rt.runner.stop()

    # ------------------------------------------------------------ 内部

    def _touch(self) -> None:
        self._last_activity = self.clock()

    def _ensure_sandbox(self) -> None:
        """首次执行时启动沙箱容器（local 模式是空包）。

        刻意不在构造/装配时启动：装配只造对象（见 assembly 的所有权表），容器该在
        「真的要跑东西」时起，空会话不该白占一个容器。
        """
        if self._rt is None or self._stack is None:
            raise RuntimeError(f"会话 {self.thread_id} 已关闭")
        if not self._sandbox_entered:
            self._stack.enter_context(self._rt.sandbox())
            self._sandbox_entered = True

    def _graph_input(self, task: str) -> tuple[Any, bool]:
        """按会话是否已有历史，决定「全新任务」还是「追加指令」。

        返回 `(graph_input, 是否发 AGENT_STARTED)`，与 CLI 的 resume 分支同一条判断：
        已有 messages 就只追加一条 HumanMessage（重传初始 messages 会被 `add_messages`
        追加成重复历史）。
        """
        if self._rt is None:
            raise RuntimeError(f"会话 {self.thread_id} 已关闭")
        if self._rt.existing_message_count() == 0:
            return self._rt.initial_input(task), True
        return self._rt.append_input(task), False

    def _start_worker(self, graph_input: Any, *, announce: bool) -> None:
        """起一个 worker 线程跑任务。**所有线程绑定都收敛在这里。**

        `bind_thread` / `bind_emitter` 是线程上下文作用域（不是会话作用域），
        审批恢复后新起的线程必须重新绑，否则它的图内 emit 会掉回模块单例 ——
        会话订阅者再也收不到任何事件，且**没有任何报错**。
        """
        t = threading.Thread(
            target=self._run,
            args=(graph_input, announce),
            name=f"codepilot-session-{self.thread_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._worker = t
        t.start()

    def _run(self, graph_input: Any, announce: bool) -> None:
        rt = self._rt
        if rt is None:  # 起线程与 close() 之间的竞态：直接收工
            return
        bind_thread(self.thread_id)
        bind_emitter(self.emitter)
        if announce:
            self.emitter.emit(EventType.AGENT_STARTED, agent="Supervisor", message="")
        try:
            result, paused = run_task(
                rt.graph, graph_input, rt.config, on_interrupt=self._park_for_approval
            )
        except BaseException as e:  # noqa: BLE001
            with self._lock:
                if self._status is SessionStatus.CLOSED:
                    return
                self._status = SessionStatus.IDLE
                self._error = f"{type(e).__name__}: {e}"
                self._touch()
            # `_error` 与事件文案都只有「类型 + 一句话」——那是给 UI 看的。给运维看的
            # 栈必须另记一处：worker 是后台线程，这里不记，栈就**永远不落任何地方**
            # （`_error` 落到用户眼里只是「KeyError: 'foo'」）。
            logger.error(
                "会话 %s 执行失败: %s", self.thread_id, self._error, exc_info=True
            )
            self.emitter.emit(
                EventType.AGENT_FAILED, agent="Supervisor", message=self._error
            )
            self._publish_status()
            return

        with self._lock:
            if self._status is SessionStatus.CLOSED:
                return
            self._error = None
            if paused:
                # 状态已在 _park_for_approval 里置好；worker 到此结束（不占线程）。
                self._touch()
                # 记在锁内：这一支 return 之后没有别的地方能记它，而「挂起」正是最该
                # 看见的生命周期事件（界面上表现为「等你点批准」）。
                logger.info("会话 %s 挂在待批准，worker 让出线程", self.thread_id)
                return
            status = result.get("status") if isinstance(result, dict) else None

        # 与 CLI 收尾对称：finished → AGENT_COMPLETED，其余 → AGENT_FAILED。
        # **先发事件、再置 idle**：反过来的话「状态已是 idle、收尾事件还没落」有一个窗口，
        # 轮询状态的客户端此刻拉事件列表会少最后一条（真机跑测试时实测踩到，事件计数
        # 时多时少）。idle 的语义应当是「跑完了且事件都发完了」。
        if status == "finished":
            self.emitter.emit(EventType.AGENT_COMPLETED, agent="Supervisor", message="")
            logger.info("会话 %s 收尾：finished", self.thread_id)
        else:
            err = (result or {}).get("error") if isinstance(result, dict) else None
            self.emitter.emit(
                EventType.AGENT_FAILED,
                agent="Supervisor",
                message=err or f"状态 {status}",
            )
            logger.warning(
                "会话 %s 收尾：%s（%s）", self.thread_id, status or "无状态", err or "无错误文本"
            )

        with self._lock:
            if self._status is SessionStatus.CLOSED:
                return  # 收尾期间被 close()：别再把它置回 idle
            self._status = SessionStatus.IDLE
            self._result = result
            # 恢复出来的旧摘要必须让位给本轮真结果，否则续跑完还在显示上次的
            self._restored_summary = None
            self._touch()
        self._publish_status()

    def _park_for_approval(self, payloads: list[dict]) -> None:
        """`run_task` 的 on_interrupt：记录待批准并**让出线程**（返回 None）。

        返回 None 而非字符串 = 「不在这里阻塞等人类」，worker 线程就此结束；人类
        批准后由 `resume()` 起新线程继续（state 在 checkpointer 里，与这个线程无关）。

        状态必须在**这里**置位：run_task 拿到 None 就立刻返回，若等 `_run` 收尾才改，
        中间存在一个「已经挂起但状态还是 running」的窗口——那期间并发指令会直接喂给
        LangGraph，静默吞掉挂起的委派。
        """
        with self._lock:
            self._status = SessionStatus.AWAITING_APPROVAL
            self._pending_approval = [dict(p) for p in payloads]
            self._touch()
        self._publish_status()
        return None

    # ------------------------------------------------------------ 回收

    def reapable(self, idle_timeout: float) -> bool:
        """是否可回收（空闲超时）。**`running` 永远不可回收。**

        跑动中的会话可能正卡在一次几十秒的 LLM 调用或 `docker exec` 上，此时把它
        `rm -f` 等于把容器从正在跑的命令下面抽走。

        P9 起 `interrupted` 也在白名单里：它同样是「死」不是「忙」（进程上次死的时候
        它就停了，本地不可能有东西在跑）。**漏掉它等于每点开一个历史会话就永久泄漏
        一个 graph + saver**——而且那条记录还在，看起来一切正常，只有内存慢慢涨。
        """
        with self._lock:
            if self._status not in (
                SessionStatus.IDLE,
                SessionStatus.AWAITING_APPROVAL,
                SessionStatus.INTERRUPTED,
            ):
                return False
            return (self.clock() - self._last_activity) >= idle_timeout

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = [
    "RestorePayload",
    "Session",
    "SessionStatus",
    "Subscriber",
    "event_payload",
    "jsonable",
    "summarize_result",
]
