"""P7: 会话注册表 —— 服务端的会话目录 + 空闲回收。

CLI 里会话的生命周期是一个 `with` 块（进程退出即结束）；服务端没有这个天然边界，
所以需要有人**持有**会话并按策略释放：容器、checkpointer、event store 监听器都是
每会话一份的资源，没人放手就是泄漏。

## 两件事分开

- `close(id)` —— 显式销毁（DELETE /sessions/{id}、服务关停）。**关闭并从目录里移除。**
- `reap_idle()` —— 空闲回收：状态是 `idle`/`awaiting_approval` 且静置超时的会话
  一并关掉。**`running` 永不回收**（见 `Session.reapable`）。

回收的会话从目录里移除 = 之后 `GET /sessions/{id}` 是 404。事件本身在 postgres
后端下已经落库（`GET /sessions/{id}/events` 从库里回放，P7.1 补）；sqlite 后端下
事件只在进程内存，会话没了就真没了——`api/app.py` 启动时会对 sqlite 明确告警。

## 并发

`create` 在锁内完成（会话构造 ≈ 60ms，且「新建会话」是低频动作），换来的是
「检查 id 是否已被占用 → 登记」这一步的原子性，不必处理占位符回滚。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from config.settings import Settings
from runtime.session import Session

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

        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._reaper: threading.Thread | None = None
        self._stop_reaper = threading.Event()

    # ------------------------------------------------------------ 目录

    def create(self, thread_id: str | None = None) -> Session:
        """新建一个会话。id 冲突抛 `SessionExistsError`。"""
        thread_id = thread_id or uuid4().hex
        with self._lock:
            if thread_id in self._sessions:
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
        return session

    def get(self, thread_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(thread_id)

    def sessions(self) -> list[Session]:
        with self._lock:
            return list(self._sessions.values())

    def snapshots(self) -> list[dict]:
        return [s.snapshot() for s in self.sessions()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, thread_id: object) -> bool:
        with self._lock:
            return thread_id in self._sessions

    # ------------------------------------------------------------ 释放

    def close(self, thread_id: str) -> bool:
        """关闭并移除一个会话。不存在返回 False。"""
        with self._lock:
            session = self._sessions.pop(thread_id, None)
        if session is None:
            return False
        session.close()
        return True

    def reap_idle(self) -> list[str]:
        """回收空闲会话，返回被回收的 id 列表。

        **`running` 的会话不在候选里**——会话可能正卡在一次几十秒的 LLM 调用或
        `docker exec` 上，此时 `rm -f` 容器会把正在跑的命令从底下抽走。

        「判可回收」与「摘出目录」在同一把锁内完成：中间那一瞬正是 worker 可能
        被唤醒去跑东西的窗口，先摘出来才能保证回收不会打到刚跑起来的会话。
        """
        reaped: list[str] = []
        for thread_id in [s.thread_id for s in self.sessions()]:
            with self._lock:
                session = self._sessions.get(thread_id)
                if session is None or not session.reapable(self.idle_timeout):
                    continue
                self._sessions.pop(thread_id, None)
            session.close()
            reaped.append(thread_id)
        return reaped

    def close_all(self) -> None:
        """关停全部会话（服务退出时用）。单个会话关闭失败不影响其余。"""
        for session in self.sessions():
            try:
                session.close()
            except Exception:  # noqa: BLE001  关停要尽力而为，不能因为一个炸掉而漏掉其余
                pass
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
                    pass

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
