"""P7-5: FastAPI 服务层 —— 真起 ASGI（TestClient 跑真 lifespan），LLM 与容器都是假的。

不打真 LLM、不起真容器、不连 PG（用默认 sqlite 后端）。这个文件钉的是 HTTP 层的
**翻译契约**（状态机 → 状态码）与三条最容易静默出错的路径：

1. **忙时 409 必须覆盖两个状态**（`running` 与 `awaiting_approval`）——挂起中放行一条
   新指令，LangGraph 会静默吞掉那个 interrupt（探针实证，见 runtime/session.py）；
2. **审批值必须精确是字符串 `"yes"`** ——`supervisor.py` 的判定是 `if answer != "yes"`，
   `{"approved": true}` 直接透传会静默变成「用户拒绝」。这里用 spy 拦 `Command` 断言实参；
3. **关停要真收干净**（会话 → 容器）。容器 `stop()` 由 `Session.close()` 显式调，
   不依赖 `with` 的退出路径（没跑过东西的会话根本不进 `rt.sandbox()`）。

WS 用例一律 **thread + join 超时**：`TestClient.receive_json()` 没有超时参数，
事件不来会永久挂死整个测试套件（连 pytest 的 timeout 插件都没装）。
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import runtime.session as session_mod
from api.app import create_app
from config.settings import Settings

# ---------------------------------------------------------------- 假件

QUICKSORT = "def quicksort(arr):\n    return sorted(arr)\n"
WS_TIMEOUT = 30.0


def _tool_call(idx, name, args):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{idx}", "type": "tool_call"}],
    )


class _ScriptedLLM:
    """按脚本吐 AIMessage。工厂每次调用都新建实例，会话之间不共享状态。"""

    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if not self.script:
            return AIMessage(content="(fallback done)")
        return self.script.pop(0)


class _GatedLLM:
    """第一次 invoke 卡在闸门上，把会话**确定性地**按在 running 状态。"""

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


class _FakeRunner:
    """假沙箱容器：只记生命周期，不碰 docker。"""

    def __init__(self):
        self.container_name = "fake-container"
        self.image = "fake:latest"
        self.entered = 0
        self.exited = 0
        self.stopped = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return None

    def run(self, command, timeout=None):
        return f"(fake) {command}"

    def stop(self):
        self.stopped += 1


# ---------------------------------------------------------------- 脚本

DONE = {"Supervisor": [AIMessage(content="任务完成：没有要改的东西。")]}

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


# ---------------------------------------------------------------- 脚手架


def _settings(tmp_path) -> Settings:
    """直接构造 Settings（不走 from_env）——本机 .env 的 PERSISTENCE_BACKEND 等不许影响测试。"""
    return Settings(
        llm_api_key="fake-key",
        llm_base_url="",
        llm_model="fake",
        workspace_root=Path(tmp_path) / "ws",
        checkpoint_db_path=Path(tmp_path) / "data" / "cp.db",
        session_idle_timeout=1800.0,
    )


def _llm_factory(scripts=None, gate=None):
    scripts = scripts if scripts is not None else DONE

    def make_llm(role):
        if gate is not None and role == "Supervisor":
            return _GatedLLM(gate, scripts.get(role, []))
        return _ScriptedLLM(scripts.get(role, []))

    return make_llm


@contextmanager
def _client(tmp_path, *, scripts=None, gate=None, runner=None, approvals=()):
    """起一个真 app（真 lifespan）+ TestClient。"""
    app = create_app(
        _settings(tmp_path),
        make_llm=_llm_factory(scripts, gate),
        runner_factory=(lambda settings, ws: runner) if runner is not None else None,
        require_approval_for=approvals,
    )
    with TestClient(app) as client:
        yield client


def _wait_until(pred, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _create(client, thread_id: str | None = None) -> str:
    response = client.post("/sessions", json={"thread_id": thread_id})
    assert response.status_code == 201, response.text
    return response.json()["thread_id"]


def _status(client, thread_id: str) -> str:
    return client.get(f"/sessions/{thread_id}").json()["status"]


def _await_status(client, thread_id: str, expected: str, timeout: float = 15.0) -> None:
    assert _wait_until(
        lambda: _status(client, thread_id) == expected, timeout
    ), f"会话没有进入 {expected}，当前 {_status(client, thread_id)}"


def _await_events_settled(client, thread_id: str, timeout: float = 15.0) -> None:
    """等到收尾事件落进会话——**要数事件条数的用例必须先经过这里**。

    `_await_status(idle)` 其实已经蕴含它（`Session._run` 现在是「先发收尾事件、再置
    idle」），这里显式等一次，免得用例把「条数」押在另一处的顺序实现上。
    """
    def settled() -> bool:
        events = client.get(f"/sessions/{thread_id}/events").json()["events"]
        return bool(events) and events[-1]["event"]["type"] in (
            "AgentCompleted",
            "AgentFailed",
        )

    assert _wait_until(settled, timeout), "事件流没有收尾（AgentCompleted/AgentFailed）"


def _start(client, thread_id: str, task: str = "写一个快排"):
    response = client.post(f"/sessions/{thread_id}/messages", json={"task": task})
    assert response.status_code == 202, response.text
    return response


# ---------------------------------------------------------------- 会话目录


def test_create_session_returns_201_with_generated_id(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/sessions", json={})
        assert response.status_code == 201
        body = response.json()
        assert body["thread_id"]
        assert body["status"] == "idle"
        assert body["event_count"] == 0
        assert body["error"] is None
        # 也接受完全不带 body（POST /sessions）
        assert client.post("/sessions").status_code == 201


def test_create_session_with_explicit_id_and_conflict(tmp_path):
    with _client(tmp_path) as client:
        assert _create(client, "my-session") == "my-session"
        again = client.post("/sessions", json={"thread_id": "my-session"})
        assert again.status_code == 409
        # 冲突的是「建」不是「找」——原会话不该被动过
        assert _status(client, "my-session") == "idle"


def test_list_sessions(tmp_path):
    with _client(tmp_path) as client:
        _create(client, "a")
        _create(client, "b")
        body = client.get("/sessions").json()
        assert sorted(s["thread_id"] for s in body["sessions"]) == ["a", "b"]


def test_unknown_session_is_404_on_every_route(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/sessions/nope").status_code == 404
        assert client.get("/sessions/nope/events").status_code == 404
        assert client.post("/sessions/nope/messages", json={"task": "x"}).status_code == 404
        assert client.post("/sessions/nope/approval", json={"approved": True}).status_code == 404
        assert client.delete("/sessions/nope").status_code == 404


def test_delete_session_is_204_then_404(tmp_path):
    with _client(tmp_path) as client:
        thread_id = _create(client)
        assert client.delete(f"/sessions/{thread_id}").status_code == 204
        assert client.get(f"/sessions/{thread_id}").status_code == 404
        assert client.delete(f"/sessions/{thread_id}").status_code == 404


def test_empty_task_is_rejected_by_validation(tmp_path):
    with _client(tmp_path) as client:
        thread_id = _create(client)
        assert client.post(f"/sessions/{thread_id}/messages", json={"task": ""}).status_code == 422


# ---------------------------------------------------------------- 动作


def test_message_runs_task_and_returns_to_idle(tmp_path):
    with _client(tmp_path) as client:
        thread_id = _create(client)
        body = _start(client, thread_id).json()
        # 立刻返回 202：任务是后台 worker 线程跑的，此刻状态已经是 running
        assert body["status"] == "running"
        _await_status(client, thread_id, "idle")
        snapshot = client.get(f"/sessions/{thread_id}").json()
        assert snapshot["result"]["status"] == "finished"
        assert snapshot["event_count"] > 0


def test_busy_while_running_is_409(tmp_path):
    gate = threading.Event()
    with _client(tmp_path, gate=gate) as client:
        thread_id = _create(client)
        _start(client, thread_id)
        _await_status(client, thread_id, "running")

        blocked = client.post(f"/sessions/{thread_id}/messages", json={"task": "再来一个"})
        assert blocked.status_code == 409
        # detail 带状态，前端据此决定显示什么（转圈 vs 审批按钮）
        assert blocked.json()["detail"]["status"] == "running"

        gate.set()
        _await_status(client, thread_id, "idle")


def test_busy_while_awaiting_approval_is_409(tmp_path):
    with _client(tmp_path, scripts=DELEGATE_TO_CODER, approvals=("coder",)) as client:
        thread_id = _create(client)
        _start(client, thread_id)
        _await_status(client, thread_id, "awaiting_approval")

        blocked = client.post(f"/sessions/{thread_id}/messages", json={"task": "另一件事"})
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["status"] == "awaiting_approval"

        assert client.post(
            f"/sessions/{thread_id}/approval", json={"approved": True}
        ).status_code == 200
        _await_status(client, thread_id, "idle")


def test_approval_without_pending_is_409(tmp_path):
    with _client(tmp_path) as client:
        thread_id = _create(client)
        response = client.post(f"/sessions/{thread_id}/approval", json={"approved": True})
        assert response.status_code == 409
        assert response.json()["detail"]["status"] == "idle"


def test_approval_must_carry_approved_field(tmp_path):
    with _client(tmp_path) as client:
        thread_id = _create(client)
        assert client.post(f"/sessions/{thread_id}/approval", json={}).status_code == 422


# ---------------------------------------------------------------- 审批值（本批最核心的断言）


def _spy_command(monkeypatch) -> list:
    """拦 `Session.resume` 造 Command 的那一步，记下**实际传进 resume 的实参**。

    直接断言这张表，而不是断言「跑完了」——「跑完了」在 True 被当成 "no" 时同样成立
    （拒绝也会正常收尾），只有实参能钉住这个静默错误。
    """
    seen: list = []
    real = session_mod.Command

    def spy(**kwargs):
        seen.append(kwargs.get("resume"))
        return real(**kwargs)

    monkeypatch.setattr(session_mod, "Command", spy)
    return seen


def test_approved_true_reaches_command_as_exactly_yes(tmp_path, monkeypatch):
    seen = _spy_command(monkeypatch)
    with _client(tmp_path, scripts=DELEGATE_TO_CODER, approvals=("coder",)) as client:
        thread_id = _create(client)
        _start(client, thread_id)
        _await_status(client, thread_id, "awaiting_approval")
        assert seen == []  # 挂起那次是 interrupt()，不经过 Command

        assert client.post(
            f"/sessions/{thread_id}/approval", json={"approved": True}
        ).status_code == 200
        assert seen == ["yes"], "resume 值必须精确是字符串 'yes'（否则静默变拒绝）"

        _await_status(client, thread_id, "idle")
        # 批准真的落地了：文件真写出来了
        assert (Path(tmp_path) / "ws" / thread_id / "main.py").read_text(encoding="utf-8") == QUICKSORT


def test_approved_false_reaches_command_as_no_and_writes_nothing(tmp_path, monkeypatch):
    seen = _spy_command(monkeypatch)
    with _client(tmp_path, scripts=DELEGATE_TO_CODER, approvals=("coder",)) as client:
        thread_id = _create(client)
        _start(client, thread_id)
        _await_status(client, thread_id, "awaiting_approval")

        assert client.post(
            f"/sessions/{thread_id}/approval", json={"approved": False}
        ).status_code == 200
        assert seen == ["no"]

        _await_status(client, thread_id, "idle")
        assert not (Path(tmp_path) / "ws" / thread_id / "main.py").exists()


# ---------------------------------------------------------------- 事件回填


def test_events_endpoint_backfills_and_filters_since(tmp_path):
    with _client(tmp_path) as client:
        thread_id = _create(client)
        _start(client, thread_id)
        _await_events_settled(client, thread_id)

        body = client.get(f"/sessions/{thread_id}/events").json()
        events = body["events"]
        assert events, "事件回填不该是空的"
        assert [e["seq"] for e in events] == list(range(len(events)))
        assert events[0]["event"]["type"] == "AgentStarted"
        assert events[-1]["event"]["type"] in ("AgentCompleted", "AgentFailed")
        assert all(e["event"]["thread_id"] == thread_id for e in events)

        # ?since= 跳过已收过的（WS 断线重连靠它补）
        tail = client.get(f"/sessions/{thread_id}/events", params={"since": 2}).json()
        assert [e["seq"] for e in tail["events"]] == [s for s in range(len(events)) if s >= 2]


# ---------------------------------------------------------------- 关停


def test_shutdown_closes_sessions_and_stops_runner(tmp_path):
    runner = _FakeRunner()
    with _client(tmp_path, runner=runner) as client:
        thread_id = _create(client)
        _start(client, thread_id)
        _await_status(client, thread_id, "idle")
        assert runner.entered == 1, "第一条指令才起容器（空会话不占容器）"
        assert runner.stopped == 0

    # 退出 TestClient 的 with = 走 lifespan 的 shutdown（registry.close_all）
    assert runner.stopped >= 1
    assert runner.exited == 1


# ---------------------------------------------------------------- WebSocket


def _in_thread(fn, timeout: float = WS_TIMEOUT) -> None:
    """跑 WS 交互。**必须带超时**：`receive_json()` 没有超时参数，事件不来就是永久挂死。

    ⚠️ 所以本文件里每个 WS 用例收到的条数都必须是**确定**的：少收一条就是
    `receive_json()` 永久阻塞。兜底只有这里的 join 超时——而超时后那个线程还卡在
    `portal.call` 上，TestClient 的 shutdown 会跟着一起卡（真机踩过一次，600s 没退出）。
    换句话说：**别写「等一条可能不来的消息」的用例**。
    """
    error: list = []

    def wrapper():
        try:
            fn()
        except BaseException as e:  # noqa: BLE001  带回主线程再抛，否则只剩一个静默的线程
            error.append(e)

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), f"WS 交互超过 {timeout}s 未结束（多半是事件没到）"
    if error:
        raise error[0]


def test_ws_streams_status_backlog_and_live_events(tmp_path):
    with _client(tmp_path) as client:
        thread_id = _create(client)
        received: list = []

        def interact():
            with client.websocket_connect(f"/sessions/{thread_id}/ws") as websocket:
                connect_status = websocket.receive_json()
                assert connect_status["kind"] == "status"
                assert connect_status["status"] == "idle"
                assert connect_status["thread_id"] == thread_id
                # 连上、订阅好之后再发指令 —— 之后的事件必须全走实时推送
                _start(client, thread_id)
                while len(received) < 40:
                    envelope = websocket.receive_json()
                    received.append(envelope)
                    event = envelope.get("event") or {}
                    if (
                        envelope["kind"] == "event"
                        and event.get("agent") == "Supervisor"
                        and event.get("type") in ("AgentCompleted", "AgentFailed")
                    ):
                        break

        _in_thread(interact)

        events = [e["event"] for e in received if e["kind"] == "event"]
        types = [e["type"] for e in events]
        assert "AgentStarted" in types
        assert types[-1] in ("AgentCompleted", "AgentFailed")
        # seq 连续无缺口（回填与实时推送不重不漏）
        seqs = [e["seq"] for e in received if e["kind"] == "event"]
        assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))
        # 推的是结构化 Event，不是 format_event 的渲染串（UI 要 agent 归属）
        assert all("agent" in e and "timestamp" in e for e in events)


def test_ws_since_skips_already_seen_events(tmp_path):
    """断线重连：带 `?since=` 只补没收到的那段，不重复回填全部历史。"""
    with _client(tmp_path) as client:
        thread_id = _create(client)
        _start(client, thread_id)
        _await_events_settled(client, thread_id)
        total = len(client.get(f"/sessions/{thread_id}/events").json()["events"])
        assert total >= 3, "事件太少，这个用例没意义（AgentStarted/AgentStep/AgentCompleted 至少 3 条）"

        seqs: list = []

        def interact():
            with client.websocket_connect(f"/sessions/{thread_id}/ws?since=1") as websocket:
                assert websocket.receive_json()["kind"] == "status"
                for _ in range(total - 1):
                    seqs.append(websocket.receive_json()["seq"])

        _in_thread(interact)
        assert seqs == list(range(1, total))


def test_ws_disconnect_leaves_no_subscriber(tmp_path):
    """对端断开后必须摘掉订阅者。

    只靠 send 是察觉不到断开的（安静会话两边都没动静）——handler 会抱着订阅者挂到天荒地老，
    推给一个已死的 queue。所以 WS handler 里并发跑了一个「读 receive 等断开」的任务。
    """
    with _client(tmp_path) as client:
        thread_id = _create(client)
        session = client.app.state.registry.get(thread_id)

        def connect_and_leave():
            with client.websocket_connect(f"/sessions/{thread_id}/ws") as websocket:
                websocket.receive_json()  # 连上就有 status，说明订阅已生效
                assert session.subscriber_count == 1

        _in_thread(connect_and_leave)
        assert _wait_until(lambda: session.subscriber_count == 0), "断开后订阅者没被摘掉"


def test_ws_unknown_session_sends_error_envelope(tmp_path):
    with _client(tmp_path) as client:
        with client.websocket_connect("/sessions/nope/ws") as websocket:
            envelope = websocket.receive_json()
            assert envelope["kind"] == "error"
            assert "nope" in envelope["message"]
