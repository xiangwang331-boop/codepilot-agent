"""P7-4: Session + Registry —— 全 fake，不起 HTTP、不起容器、不连网络。

这个文件钉的是服务端最容易「静默出错」的几处：
- 会话忙时**必须**拒绝新指令（LangGraph 会静默吞掉挂起中的 interrupt，探针实证）；
- 审批让出线程 → resume 起新线程 → 图跑完且文件真落盘；
- 订阅回填与实时推送**既不漏也不重**；
- 多会话并发**互不串台**（各自 emitter / workspace / 图）；
- 空闲回收只碰 `idle`/`awaiting_approval`，**绝不碰 `running`**。

并发用例刻意用**本文件的**脚本 LLM，不用 `conftest.FakeLLM`
（后者把 `script`/`calls` 挂在实例上，并发下自己就串台）。
"""
from __future__ import annotations

import gc
import threading
import time
import weakref
from collections import deque
from pathlib import Path

from langchain_core.messages import AIMessage

from config.settings import Settings
from events.events import EventType
from runtime.registry import SessionExistsError, SessionRegistry
from runtime.session import Session, SessionStatus
from tools.command_runner import DockerCommandRunner

# ---------------------------------------------------------------- 假件

QUICKSORT = "def quicksort(arr):\n    return sorted(arr)\n"


def _tool_call(idx, name, args):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{idx}", "type": "tool_call"}],
    )


class _ScriptedLLM:
    """按脚本吐 AIMessage。工厂每次调用都新建实例，所以并发会话互不共享状态。"""

    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if not self.script:
            return AIMessage(content="(fallback done)")
        return self.script.pop(0)


class _GatedLLM:
    """第一次 invoke 卡在闸门上，用来把会话**确定性地**按在 running 状态。"""

    def __init__(self, gate: threading.Event, script):
        self.gate = gate
        self.script = list(script)
        self.entered = threading.Event()

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.entered.set()
        assert self.gate.wait(timeout=10), "闸门没被放开"
        if not self.script:
            return AIMessage(content="(fallback done)")
        return self.script.pop(0)


class _Clock:
    """可控时钟（回收策略不该靠 sleep 去测）。"""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------- 脚手架

DELEGATE_TO_CODER = {
    "Supervisor": [
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写一个 quicksort 到 main.py"}),
        AIMessage(content="完成：Coder 已写好快排。"),
    ],
    "coder": [
        _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT}),
        AIMessage(content="快排已写入 main.py"),
    ],
}


def _settings(tmp_path) -> Settings:
    return Settings(
        llm_api_key="fake-key",
        llm_base_url="",
        llm_model="fake",
        workspace_root=Path(tmp_path) / "ws",
        checkpoint_db_path=Path(tmp_path) / "data" / "cp.db",
    )


def _llm_factory(scripts=None):
    scripts = scripts if scripts is not None else DELEGATE_TO_CODER

    def make_llm(role):
        return _ScriptedLLM(scripts.get(role, []))

    return make_llm


def _wait_until(pred, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _open(tmp_path, **kwargs) -> Session:
    """起一个会话。默认**关掉审批**（`require_approval_for=()`）——需要挂起的用例显式打开。"""
    kwargs.setdefault("make_llm", _llm_factory())
    kwargs.setdefault("require_approval_for", ())
    return Session(
        kwargs.pop("thread_id", "sess-1"),
        kwargs.pop("settings", None) or _settings(tmp_path),
        workspace_root=kwargs.pop("workspace_root", Path(tmp_path) / "ws" / "sess-1"),
        **kwargs,
    )


def _workspace_file(tmp_path, name="main.py", thread_id="sess-1") -> Path:
    return Path(tmp_path) / "ws" / thread_id / name


# ---------------------------------------------------------------- 状态机


def test_begin_rejected_while_awaiting_approval(tmp_path):
    """**P7 第一号坑的回归**：挂起中发新指令必须被拒，而不是让 LangGraph 静默吞掉。

    探针实证：挂起时再用普通 input invoke 一次，不报错，但 `next` 清空、
    `interrupts` 清零，挂起的委派**被丢弃且文件从未创建**。所以这里不仅断言
    `begin()` 返回 False，还要断言批准后文件**真的落盘了**——若中途被吞，
    文件不存在，这条断言就会挂。
    """
    with _open(tmp_path, require_approval_for=("coder",)) as s:
        assert s.begin("写快排") is True
        assert _wait_until(lambda: s.status is SessionStatus.AWAITING_APPROVAL)
        assert s.pending_approval, "挂起时必须留下待批准内容给 UI"

        # 挂着的时候再发指令 —— 必须被拒
        assert s.begin("再写一个别的") is False
        assert s.status is SessionStatus.AWAITING_APPROVAL
        assert s.pending_approval, "被拒的指令不该清掉待批准状态"

        # 批准 → 继续跑完 → 文件真落盘（证明挂起的委派没被吞）
        assert s.resume(True) is True
        assert _wait_until(lambda: s.status is SessionStatus.IDLE)
        assert _workspace_file(tmp_path).read_text(encoding="utf-8") == QUICKSORT


def test_begin_rejected_while_running(tmp_path):
    """`running` 同样挡新指令（两个状态都要挡，不能只看线程活没活着）。"""
    gate = threading.Event()
    gated: list[_GatedLLM] = []

    def make_llm(role):
        if role == "Supervisor":
            llm = _GatedLLM(gate, DELEGATE_TO_CODER["Supervisor"])
            gated.append(llm)
            return llm
        return _ScriptedLLM(DELEGATE_TO_CODER.get(role, []))

    with _open(tmp_path, make_llm=make_llm) as s:
        assert s.begin("写快排") is True
        try:
            assert gated and gated[0].entered.wait(timeout=10), "LLM 没被调到"
            assert s.status is SessionStatus.RUNNING

            assert s.begin("插一条") is False
            assert s.resume(True) is False, "没挂起时 resume 必须被拒"
        finally:
            gate.set()
        assert _wait_until(lambda: s.status is SessionStatus.IDLE)


def test_resume_only_valid_when_awaiting(tmp_path):
    with _open(tmp_path, require_approval_for=("coder",)) as s:
        assert s.resume(True) is False  # 还没开始跑
        assert s.begin("写快排") is True
        assert _wait_until(lambda: s.status is SessionStatus.AWAITING_APPROVAL)
        assert s.resume(True) is True
        assert _wait_until(lambda: s.status is SessionStatus.IDLE)
        assert s.resume(True) is False  # 已经跑完了

        s.close()
        assert s.status is SessionStatus.CLOSED
        assert s.begin("再跑") is False  # 关掉的会话不能再开始
        s.close()  # 幂等


def test_yield_and_resume_runs_on_a_new_worker_thread(tmp_path):
    """审批让出线程：worker 结束 → resume 起**新**线程 → 图跑完且文件落盘。"""
    with _open(tmp_path, require_approval_for=("coder",)) as s:
        assert s.begin("写快排") is True
        assert _wait_until(lambda: s.status is SessionStatus.AWAITING_APPROVAL)
        parked_worker = s._worker
        assert not parked_worker.is_alive(), "挂起后 worker 线程必须结束（不占线程）"

        assert s.resume(True) is True
        assert _wait_until(lambda: s.status is SessionStatus.IDLE)
        assert s._worker is not parked_worker, "恢复必须起新线程"
        assert not s._worker.is_alive()
        assert _workspace_file(tmp_path).exists()


def test_rejection_returns_error_without_writing_file(tmp_path):
    with _open(tmp_path, require_approval_for=("coder",)) as s:
        assert s.begin("写快排") is True
        assert _wait_until(lambda: s.status is SessionStatus.AWAITING_APPROVAL)
        assert s.resume(False) is True
        assert _wait_until(lambda: s.status is SessionStatus.IDLE)
        assert not _workspace_file(tmp_path).exists()
        # 拒绝后 supervisor 拿到 ERROR ToolMessage 继续决策，最终正常收尾
        assert s.snapshot()["result"]["status"] == "finished"


def test_resume_normalizes_bool_to_exact_yes(tmp_path):
    """`Command(resume=True)` 会被 `answer != "yes"` 静默当成拒绝 —— 必须归一化成 "yes"。"""
    with _open(tmp_path, require_approval_for=("coder",)) as s:
        assert s.begin("写快排") is True
        assert _wait_until(lambda: s.status is SessionStatus.AWAITING_APPROVAL)
        # 传的是 API 那种布尔；没归一化的话委派被静默丢弃、文件不会出现
        assert s.resume(True) is True
        assert _wait_until(lambda: s.status is SessionStatus.IDLE)
        assert _workspace_file(tmp_path).exists()


def test_second_instruction_appends_to_existing_session(tmp_path):
    """已有历史的会话再 begin：追加一条 HumanMessage，而不是重传初始 state。"""
    scripts = {
        "Supervisor": [
            _tool_call(1, "delegate", {"specialist": "coder", "task": "写 main.py"}),
            AIMessage(content="第一轮完成。"),
            AIMessage(content="第二轮完成。"),
        ],
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT}),
            AIMessage(content="写好了"),
        ],
    }
    with _open(tmp_path, make_llm=_llm_factory(scripts)) as s:
        assert s.begin("写快排") is True
        assert _wait_until(lambda: s.status is SessionStatus.IDLE)
        assert s.begin("再检查一遍") is True
        assert _wait_until(lambda: s.status is SessionStatus.IDLE)
        assert s.snapshot()["result"]["result"] == "第二轮完成。"
        # 没有重复的初始 SystemMessage（重传初始 messages 会被 add_messages 追加）
        msgs = s._rt.graph.get_state(s._rt.config).values["messages"]
        assert sum(1 for m in msgs if m.type == "system") == 1


# ---------------------------------------------------------------- 事件与订阅


def test_subscribe_backfills_then_streams_live(tmp_path):
    with _open(tmp_path) as s:
        received: list[dict] = []
        status_env, backlog = s.subscribe(received.append)
        assert status_env["kind"] == "status"
        assert status_env["status"] == "idle"
        assert backlog == []

        assert s.begin("写快排") is True
        # 等 **Supervisor** 的收尾事件，不是随便一个 AgentCompleted —— 子图跑完
        # 也会发一条（agent='coder'），等错了会拿到半截流。
        assert _wait_until(
            lambda: any(
                e.get("event", {}).get("type") == "AgentCompleted"
                and e["event"]["agent"] == "Supervisor"
                for e in received
            )
        )

        live = [e for e in received if e["kind"] == "event"]
        # seq 连续（不重）+ 条数与会话事件一致（不漏）
        assert [e["seq"] for e in live] == list(range(len(live)))
        assert len(live) == len(s.events)
        assert live[0]["event"]["type"] == "AgentStarted"
        assert live[-1]["event"]["type"] == "AgentCompleted"
        # 推的是结构化事件（不是 format_event 的渲染结果）
        assert live[0]["event"]["agent"] == "Supervisor"
        assert live[0]["event"]["thread_id"] == s.thread_id

        # 断线重连：再来一个订阅者，能拿到整段回填
        again: list[dict] = []
        _, replay = s.subscribe(again.append)
        assert [e["seq"] for e in replay] == list(range(len(replay)))
        assert [e["event"]["type"] for e in replay] == [
            e["event"]["type"] for e in live
        ]


def test_subscribe_while_emitting_has_no_gap_and_no_duplicate(tmp_path):
    """回填与实时推送的竞态：另一个线程正密集 emit 时注册订阅者。

    断言「并集恰好等于全体、且同一序号不出现两次」——漏了会缺号，重了会见到
    同一个 seq 两次。
    """
    with _open(tmp_path) as s:
        stop = threading.Event()

        def pump():
            while not stop.is_set():
                s.emitter.emit(EventType.AGENT_STEP, agent="pump", message="tick")

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            time.sleep(0.05)  # 让它先跑热
            received: list[dict] = []
            _, backlog = s.subscribe(received.append)
            backfilled = {e["seq"] for e in backlog}
            time.sleep(0.05)
        finally:
            stop.set()
            t.join(timeout=5)

        live = [e["seq"] for e in received if e["kind"] == "event"]
        assert len(live) == len(set(live)), "同一事件被推了两次（重复投递）"
        assert not (set(backfilled) & set(live)), "回填与实时推送重叠"
        # 订阅之后产生的事件必须全部从实时通道到达（没有缺口）
        assert set(range(len(s.events))) - backfilled <= set(live)


def test_six_concurrent_sessions_do_not_crosstalk(tmp_path):
    """6 个会话并发跑：事件归属各自 emitter、workspace 各自独立。"""
    settings = _settings(tmp_path)
    sessions = []
    try:
        for i in range(6):
            sessions.append(
                Session(
                    f"sess-{i}",
                    settings,
                    workspace_root=Path(tmp_path) / "ws" / f"sess-{i}",
                    make_llm=_llm_factory(),
                    require_approval_for=(),  # 并发用例不引入审批交互
                )
            )

        for s in sessions:
            assert s.begin("写快排") is True
        for s in sessions:
            assert _wait_until(lambda s=s: s.status is SessionStatus.IDLE), s.thread_id

        for s in sessions:
            assert s.events, "每个会话都该有自己的事件"
            assert all(e.thread_id == s.thread_id for e in s.events)
            assert {p.name for p in (Path(tmp_path) / "ws" / s.thread_id).iterdir()} == {
                "main.py"
            }
            assert s.snapshot()["result"]["result"] == "完成：Coder 已写好快排。"
    finally:
        for s in sessions:
            s.close()


# ---------------------------------------------------------------- Registry


def test_registry_create_get_close(tmp_path):
    settings = _settings(tmp_path)
    with SessionRegistry(settings, make_llm=_llm_factory()) as reg:
        s = reg.create("a")
        assert reg.get("a") is s
        assert "a" in reg and len(reg) == 1
        assert reg.create().thread_id  # 自动生成 id
        assert reg.get("nope") is None

        assert reg.close("a") is True
        assert reg.get("a") is None
        assert reg.close("a") is False  # 已移除

        reg.create("b")
        try:
            reg.create("b")
        except SessionExistsError:
            pass
        else:  # pragma: no cover
            raise AssertionError("重复 id 必须抛 SessionExistsError")


def test_registry_workspace_is_per_session(tmp_path):
    settings = _settings(tmp_path)
    with SessionRegistry(settings, make_llm=_llm_factory()) as reg:
        a, b = reg.create("a"), reg.create("b")
        assert a.workspace_root != b.workspace_root
        assert a.workspace_root == Path(tmp_path) / "ws" / "a"


def test_reaper_reclaims_idle_and_awaiting_but_never_running(tmp_path):
    clock = _Clock()
    gate = threading.Event()
    # make_llm 在 **Session 构造时**被调（建图在 __init__ 里）→ 消费顺序 = create 顺序：
    # idle 用不到 LLM、parked 要能真挂起（普通脚本）、running 要定在闸门上。
    sup_factories = deque(
        [
            lambda: _ScriptedLLM([]),
            lambda: _ScriptedLLM(DELEGATE_TO_CODER["Supervisor"]),
            lambda: _GatedLLM(gate, DELEGATE_TO_CODER["Supervisor"]),
        ]
    )

    def make_llm(role):
        if role == "Supervisor":
            return sup_factories.popleft()()
        return _ScriptedLLM(DELEGATE_TO_CODER.get(role, []))

    reg = SessionRegistry(
        _settings(tmp_path), make_llm=make_llm, idle_timeout=100.0, clock=clock
    )
    try:
        idle = reg.create("idle")
        parked = reg.create("parked")
        running = reg.create("running")

        assert parked.begin("写快排") is True
        assert _wait_until(lambda: parked.status is SessionStatus.AWAITING_APPROVAL)
        assert running.begin("写快排") is True
        try:
            assert _wait_until(lambda: running.status is SessionStatus.RUNNING)
            assert running.status is SessionStatus.RUNNING  # 闸门没开，它就一直跑着

            clock.advance(200.0)  # 全部越过空闲阈值
            reaped = reg.reap_idle()

            assert "idle" in reaped and "parked" in reaped
            assert "running" not in reaped, "running 会话绝不能回收（容器/命令正在用）"
            assert running.status is SessionStatus.RUNNING
            assert "idle" not in reg and "parked" not in reg
            assert reg.get("running") is running
        finally:
            gate.set()
        assert _wait_until(lambda: running.status is not SessionStatus.RUNNING)
    finally:
        reg.close_all()


def test_reap_respects_idle_timeout(tmp_path):
    clock = _Clock()
    reg = SessionRegistry(
        _settings(tmp_path), make_llm=_llm_factory(), idle_timeout=100.0, clock=clock
    )
    try:
        reg.create("fresh")
        clock.advance(99.0)
        assert reg.reap_idle() == []
        clock.advance(1.0)
        assert reg.reap_idle() == ["fresh"]
    finally:
        reg.close_all()


def test_reaper_thread_reclaims_and_stops(tmp_path):
    clock = _Clock()
    reg = SessionRegistry(
        _settings(tmp_path), make_llm=_llm_factory(), idle_timeout=0.0, clock=clock
    )
    try:
        reg.create("gone")
        reg.start_reaper(interval=0.01)
        assert _wait_until(lambda: "gone" not in reg, timeout=5)
    finally:
        reg.stop_reaper()
        reg.close_all()


def test_close_all_closes_every_session(tmp_path):
    reg = SessionRegistry(_settings(tmp_path), make_llm=_llm_factory())
    sessions = [reg.create(f"s{i}") for i in range(3)]
    reg.close_all()
    assert len(reg) == 0
    assert all(s.status is SessionStatus.CLOSED for s in sessions)


# ---------------------------------------------------------------- runner 生命周期


def test_stop_releases_atexit_hold_on_runner(tmp_path):
    """`atexit.register(self.stop)` 让 runner 永不回收 —— stop() 必须摘掉它。

    服务端每会话一个容器实例，不摘就是持续泄漏（对象连带 workspace 路径、镜像名
    一直挂着）。没启动过的 runner 也要摘：atexit 的强引用在**构造**时就挂上了。
    """
    runner = DockerCommandRunner(tmp_path / "ws", container_name="ctr-gc")
    ref = weakref.ref(runner)
    runner.stop()
    del runner
    gc.collect()
    assert ref() is None


def test_session_close_releases_runner(tmp_path):
    """会话关闭后 runner 可被回收（沙箱不随会话对象一起泄漏）。"""
    runner = DockerCommandRunner(tmp_path / "ws", container_name="ctr-sess")
    ref = weakref.ref(runner)
    with _open(tmp_path, runner_factory=lambda settings, ws: runner) as s:
        assert s.runner is runner
    del s, runner
    gc.collect()
    assert ref() is None
