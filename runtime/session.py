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
```

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
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from langgraph.types import Command

from config.settings import Settings
from events.events import Event, EventEmitter, EventType, bind_emitter, bind_thread
from runtime.assembly import SessionRuntime, build_runtime
from runtime.driver import normalize_answer, run_task

# 订阅者签名：收一个 WS 信封字典（见模块 docstring）。
Subscriber = Callable[[dict], None]


class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    CLOSED = "closed"


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
        self._events: list[Event] = []
        self._pending_approval: list[dict] = []
        self._worker: threading.Thread | None = None
        self._result: dict | None = None
        self._error: str | None = None
        self._sandbox_entered = False
        self._last_activity = clock()

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
        """图 state 的紧凑摘要 —— **刻意不是整包 state**，两个理由都是真机踩出来的：

        1. state 里的 `messages` 是 LangChain 消息对象。WS 走的是 Starlette 的裸
           `json.dumps` → `TypeError` → 异常被订阅路径吞掉 → **socket 静默死掉**
           （回填一个跑过任务的会话，客户端一个字都收不到，服务端没日志）；
        2. status 信封**每次状态变化都推一次**，塞整段历史会随会话无限膨胀
           （messages 里还有 2KB 的 supervisor system prompt）。

        完整历史看事件流：`GET /sessions/{id}/events` 或 WS 回填。
        """
        if not isinstance(self._result, dict):
            return None
        messages = self._result.get("messages") or []
        return {
            "status": self._result.get("status"),
            "result": jsonable(self._result.get("result")),
            "error": jsonable(self._result.get("error")),
            "iteration_count": self._result.get("iteration_count"),
            "message_count": len(messages),
        }

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
                pass

    def _publish_status(self) -> None:
        """状态信封是快照、天然幂等，没有判重问题，所有订阅者都收。"""
        try:
            self._publish({"kind": "status", **self.snapshot()})
        except Exception:  # noqa: BLE001
            pass

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
            raise

        self._publish_status()
        self._start_worker(graph_input, announce=announce)
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

        self._publish_status()
        self._start_worker(Command(resume=normalize_answer(answer)), announce=False)
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
                return
            status = result.get("status") if isinstance(result, dict) else None

        # 与 CLI 收尾对称：finished → AGENT_COMPLETED，其余 → AGENT_FAILED。
        # **先发事件、再置 idle**：反过来的话「状态已是 idle、收尾事件还没落」有一个窗口，
        # 轮询状态的客户端此刻拉事件列表会少最后一条（真机跑测试时实测踩到，事件计数
        # 时多时少）。idle 的语义应当是「跑完了且事件都发完了」。
        if status == "finished":
            self.emitter.emit(EventType.AGENT_COMPLETED, agent="Supervisor", message="")
        else:
            err = (result or {}).get("error") if isinstance(result, dict) else None
            self.emitter.emit(
                EventType.AGENT_FAILED,
                agent="Supervisor",
                message=err or f"状态 {status}",
            )

        with self._lock:
            if self._status is SessionStatus.CLOSED:
                return  # 收尾期间被 close()：别再把它置回 idle
            self._status = SessionStatus.IDLE
            self._result = result
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
        """
        with self._lock:
            if self._status not in (SessionStatus.IDLE, SessionStatus.AWAITING_APPROVAL):
                return False
            return (self.clock() - self._last_activity) >= idle_timeout

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = [
    "Session",
    "SessionStatus",
    "Subscriber",
    "event_payload",
    "jsonable",
]
