"""P8: 前端依赖的 JSON 契约 —— 把 `web/src/**` 依赖的每一个字段形状钉死在后端侧。

这个文件**不重复** `test_api.py` 的状态机断言（那边已经覆盖建/发/批/删的语义）。
这里只回答一个问题：*前端能从这些 JSON 里拿到它需要的东西吗？*

四类断言：

1. **信封与事件的字段名**，逐字对齐 `web/src/api/types.ts`；
2. **事件发射顺序**（委派嵌套、挂起重跑、拒绝闭合）—— 这是 `web/src/model/events.ts`
   折叠算法的全部依据，也是本文件最有价值的部分；
3. **错误体的形状**（404 字符串 / 409 对象 / 422 数组），对齐 `web/src/api/client.ts`
   的归一化分支；
4. **产出文件面板的依据** —— `write_file` 的 `ToolCallStarted.detail.args`
   必须带 `path` + 完整 `content`。

顺带把跑出来的**真实事件流**写成前端测试夹具（`web/src/model/__fixtures__/*.json`），让折叠
算法对着真实发射顺序验证，而不是对着谁推演的顺序。夹具里的 thread_id 与时间戳做了归一化、
并补上 WS 帧的 `kind: "event"`（见 `_normalize`），保证可重复、diff 友好。共三份：

| 夹具 | 场景 | 覆盖的折叠分支 |
|---|---|---|
| `approval-approved.json` | 批准 → coder 真跑了、写了文件 | `completed` + `attempts === 2` |
| `approval-rejected.json` | 拒绝 → 子 agent 从未启动 | `not_executed` + `childRuns === 0` |
| `batch-two-delegates.json` | 一批两个委派、只批第二个 | 悬空开块 + 子事件整段重放 |

夹具由 `test_real_stream_fixtures_are_written` / `test_batch_fixture_is_written` **每次运行都
重写**，所以它们天然跟着后端走：发射顺序一变，夹具就变，前端单测立刻红 —— 这正是想要的耦合方向
（不是让夹具冻结成一份会过期的快照）。

## 为什么用 `GET /sessions/{id}/events` 而不是 WS 抓流

WS 的 `receive_json()` **没有超时参数**，少收一条就是永久挂死。
REST 回填拿的是同一份 `session.event_envelopes()`，形状逐字相同（`{seq, event}`），
但没有阻塞风险。WS 本身另有专门的形状断言（`test_ws_envelopes_match_types_ts`），条数确定。

## 为什么复用 `test_api.py` 的私有假件

`pyproject.toml` 设了 `pythonpath = ["."]` 且 `tests/` 没有 `__init__.py`，所以
`from test_api import ...` 直接可用。重造一套脚本 LLM + 假容器只会多一份会漂移的副本。
"""
from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from api.app import create_app

from test_api import (  # noqa: E402  (依赖 pytest 的 pythonpath 注入)
    DELEGATE_TO_CODER,
    _ScriptedLLM,
    _client,
    _create,
    _settings,
    _start,
    _status,
    _tool_call,
    _wait_until,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent / "web" / "src" / "model" / "__fixtures__"
)
APPROVED_FIXTURE = FIXTURE_DIR / "approval-approved.json"
REJECTED_FIXTURE = FIXTURE_DIR / "approval-rejected.json"
BATCH_FIXTURE = FIXTURE_DIR / "batch-two-delegates.json"


def _batch_script(specs: list[tuple[str, str]]) -> list:
    """一批里放 N 个 `delegate` tool_call（`_tool_call` 只造一个，这里手写）。"""
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "delegate",
                    "args": {"specialist": specialist, "task": f"写 {path}"},
                    "id": f"call_{i}",
                    "type": "tool_call",
                }
                for i, (specialist, path) in enumerate(specs)
            ],
        ),
        AIMessage(content="两件都做完了。"),
    ]


# 一批两个委派，**只有第二个需要批准** —— 一次 interrupt，能正常跑完。
# 这是前端折叠算法要正确处理的形状：第二轮重跑会把**第一个**委派连同它的子事件
# 整段重放（`S(tester) C(tester) S(coder) C(coder)` 两遍）。
BATCH_TESTER_THEN_CODER = _batch_script([("tester", "a.py"), ("coder", "b.py")])

# 一批两个委派，**两个都需要批准** —— 一次节点执行里出现 2 个 interrupt。
# ⚠️ 这个形状目前会把第二个委派**静默丢弃**（后端缺陷，见
# `test_multiple_interrupts_in_one_batch_drop_later_delegates`）。
BATCH_CODER_TWICE = _batch_script([("coder", "a.py"), ("coder", "b.py")])


class _TaskAwareLLM:
    """按任务文本决定写哪个文件的假 LLM —— 让多委派夹具读起来是人话。

    两个理由不用 `_ScriptedLLM`：

    1. 它不看 messages，两个委派会写出同一个文件，夹具全是噪音；
    2. **子图会被 invoke 几次是不确定的**（每轮 interrupt 重跑都会把已批准的委派
       重新执行一遍），把脚本长度押在轮数上必然写错。

    这里改成「看本次子图跑的 messages 里有没有 ToolMessage」判断该收尾了 ——
    与真实 LLM 的行为一致，且与轮数无关。子图每次都是**全新 thread_id**
    （`agent/supervisor.py:103` 的 `delegate-N`），所以状态天然干净。
    """

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        task = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                task = str(m.content)
        match = re.search(r"[\w./-]+\.\w+", task)
        path = match.group(0) if match else "unknown.txt"
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content=f"{path} 写好了")
        return _tool_call(1, "write_file", {"path": path, "content": f"# {path}\n"})

# `web/src/api/types.ts` 的 AgentEvent 全部字段
EVENT_FIELDS = {"type", "agent", "message", "detail", "timestamp", "thread_id", "step", "node"}

# `web/src/api/types.ts` 的 SessionInfo 全部字段
SESSION_FIELDS = {"thread_id", "status", "approval", "result", "error", "event_count"}

# 会话里的任务文本与 coder 会写下的内容（跟着 `DELEGATE_TO_CODER` 走）
TASK = "写一个 quicksort 到 main.py"
QUICKSORT_BODY = "def quicksort(arr):\n    return sorted(arr)\n"


# ---------------------------------------------------------------- 脚手架


@contextmanager
def _flow_client(tmp_path, **kwargs):
    """跑委派会话用的 client：脚本化 supervisor + coder，且只对 coder 要求批准。"""
    with _client(tmp_path, scripts=DELEGATE_TO_CODER, approvals=("coder",), **kwargs) as client:
        yield client


def _events(client: TestClient, thread_id: str, since: int = 0) -> list[dict]:
    """取回填事件（`[{seq, event}, …]`）。"""
    response = client.get(f"/sessions/{thread_id}/events?since={since}")
    assert response.status_code == 200, response.text
    return response.json()["events"]


def _raw(client: TestClient, thread_id: str) -> list:
    """`(seq, type, agent, message)` 四元组 —— 断言顺序时比裸 dict 好读。"""
    return [
        (e["seq"], e["event"]["type"], e["event"]["agent"], e["event"]["message"])
        for e in _events(client, thread_id)
    ]


def _index_of(rows: list, pred) -> int:
    for i, row in enumerate(rows):
        if pred(row):
            return i
    raise AssertionError("没找到满足条件的事件。全部事件：\n" + "\n".join(map(str, rows)))


def _in_thread(fn, timeout: float = 15.0) -> None:
    """跑 WS 交互。**必须带超时**（`receive_json()` 没有超时参数，见模块 docstring）。"""
    error: list = []

    def wrapper():
        try:
            fn()
        except BaseException as e:  # noqa: BLE001
            error.append(e)

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    t.join(timeout)
    if error:
        raise error[0]
    assert not t.is_alive(), "WS 交互超时（事件没来）"


def _run_approval_flow(client: TestClient, *, approve: bool = True) -> list[dict]:
    """跑「委派 coder → 挂起 → 批准/拒绝 → 收尾」全程，返回全量事件信封。

    零 LLM 成本：`DELEGATE_TO_CODER` 是脚本化的 AIMessage，`_FakeRunner` 不碰 docker。
    """
    thread_id = _create(client)
    _start(client, thread_id, TASK)

    assert _wait_until(
        lambda: client.get(f"/sessions/{thread_id}").json()["status"] == "awaiting_approval"
    ), "会话没有进入待批准状态"

    response = client.post(f"/sessions/{thread_id}/approval", json={"approved": approve})
    assert response.status_code == 200, response.text

    assert _wait_until(
        lambda: client.get(f"/sessions/{thread_id}").json()["status"] == "idle"
    ), "审批后会话没有回到 idle"

    return _events(client, thread_id)


def _normalize(events: list[dict]) -> list[dict]:
    """抹掉不确定的部分，让夹具可重复、不产生无意义的 git diff。

    thread_id 换成占位符、timestamp 换成按序号递增的固定串。
    **`step` / `node` / `message` / `detail` 一律不动** —— 它们正是折叠算法要验的东西。

    另外**补上 `kind: "event"`**：REST 回填行只有 `{seq, event}`（见
    `test_event_fields_match_types_ts`），而夹具是喂给前端 `model/` 层的输入，
    前端在真实路径上拿到的是 WS 帧 `{kind:"event", seq, event}`。不补的话
    `__fixtures__/index.ts` 那个 `EventEnvelope[]` 断言就少一个必需字段 ——
    类型上说得通（`as unknown as` 抹掉了），但**假数据不能比真数据少字段**。
    """
    out = []
    for env in events:
        event = dict(env["event"])
        event["thread_id"] = "<thread>"
        event["timestamp"] = f"2026-01-01T00:00:{env['seq']:02d}+00:00"
        out.append({"kind": "event", "seq": env["seq"], "event": event})
    return out


def _write_fixture(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (
            "由 tests/test_web_ui_contract.py 从真实事件流生成 —— 不要手改。"
            "thread_id 与 timestamp 已归一化，其余字段逐字来自后端。"
        ),
        "events": events,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------- 1. 字段名


def test_session_info_fields_match_types_ts(tmp_path):
    """`SessionInfo` 的字段名必须与 `web/src/api/types.ts` 逐字一致。"""
    with _client(tmp_path) as client:
        thread_id = _create(client)
        body = client.get(f"/sessions/{thread_id}").json()
        assert set(body) == SESSION_FIELDS, f"多/少字段: {set(body) ^ SESSION_FIELDS}"
        assert isinstance(body["approval"], list)
        assert isinstance(body["event_count"], int)
        # 列表接口包了一层 {sessions: [...]}
        assert "sessions" in client.get("/sessions").json()


def test_event_fields_match_types_ts(tmp_path):
    """`AgentEvent` 的字段名必须与 `web/src/api/types.ts` 逐字一致。"""
    with _flow_client(tmp_path) as client:
        events = _run_approval_flow(client)
        assert events, "没有事件"
        for env in events:
            assert set(env) == {"seq", "event"}, set(env)
            assert set(env["event"]) == EVENT_FIELDS, set(env["event"]) ^ EVENT_FIELDS


def test_ws_envelopes_match_types_ts(tmp_path):
    """WS 三种信封的 kind 与字段，对齐 `types.ts` 的 Envelope 联合。"""
    with _client(tmp_path) as client:
        thread_id = _create(client)

        def status_envelope() -> None:
            with client.websocket_connect(f"/sessions/{thread_id}/ws") as ws:
                # 条数确定：后端 accept 后立刻推一条 status
                status = ws.receive_json()
                assert status["kind"] == "status"
                assert set(status) == SESSION_FIELDS | {"kind"}, set(status) ^ SESSION_FIELDS

        _in_thread(status_envelope)

        def unknown_session() -> None:
            from api.ws import WS_CLOSE_NO_SESSION

            with client.websocket_connect("/sessions/nope/ws") as ws:
                err = ws.receive_json()
                assert err["kind"] == "error"
                assert set(err) == {"kind", "message"}
                # 关闭帧的 code 在 receive() 的 dict 里（`WebSocketTestSession`
                # 没有 close_code 属性）
                closed = ws.receive()
                assert closed["type"] == "websocket.close"
                assert closed["code"] == WS_CLOSE_NO_SESSION

        _in_thread(unknown_session)


# ---------------------------------------------------------------- 2. 事件顺序


def test_delegation_nesting_order(tmp_path):
    """委派嵌套顺序 —— `model/events.ts` 折叠算法的直接依据。

    `ToolCallStarted(delegate)` 开块 → 子 agent 的事件在块内 → 子 `AgentCompleted`
    **先于** delegate 的 `ToolCallCompleted`（因为收口是在 `core.py` 的 `tool.invoke`
    返回之后才发的，而那次 invoke 里跑完了整段子图）。
    """
    with _flow_client(tmp_path) as client:
        thread_id = _create(client)
        _start(client, thread_id, TASK)
        assert _wait_until(
            lambda: client.get(f"/sessions/{thread_id}").json()["status"] == "awaiting_approval"
        )
        client.post(f"/sessions/{thread_id}/approval", json={"approved": True})
        assert _wait_until(
            lambda: client.get(f"/sessions/{thread_id}").json()["status"] == "idle"
        )
        rows = _raw(client, thread_id)

        i_open = _index_of(rows, lambda r: r[1] == "ToolCallStarted" and r[3] == "delegate")
        i_child = _index_of(rows, lambda r: r[1] == "AgentStarted" and r[2] == "coder")
        i_write = _index_of(rows, lambda r: r[1] == "ToolCallStarted" and r[3] == "write_file")
        i_done = _index_of(rows, lambda r: r[1] == "AgentCompleted" and r[2] == "coder")
        i_close = _index_of(
            rows, lambda r: r[1] == "ToolCallCompleted" and r[3].startswith("delegate(")
        )

        assert i_open < i_child < i_write < i_done < i_close, (
            "嵌套顺序不对：delegate 开块 → coder AgentStarted → write_file → "
            "coder AgentCompleted → delegate 收块"
        )

        # **子 agent 的 AgentStarted 全程只出现一次** —— 挂起重跑重复的是
        # ToolCallStarted(delegate)，不是这个（interrupt 在 `delegate()` 里、子 agent
        # 启动之前抛出，所以那份子图隔离根本没被进入）。折叠算法若按「第二个
        # AgentStarted」识别重跑就会完全失效。
        child_starts = [r for r in rows if r[1] == "AgentStarted" and r[2] == "coder"]
        assert len(child_starts) == 1, f"coder AgentStarted 出现了 {len(child_starts)} 次"

        # 重复的是 delegate 的 ToolCallStarted
        opens = [r for r in rows if r[1] == "ToolCallStarted" and r[3] == "delegate"]
        assert len(opens) == 2, f"期望挂起重跑产生 2 次开块，实际 {len(opens)} 次"


def test_rerun_open_block_is_byte_identical(tmp_path):
    """重跑的 `ToolCallStarted(delegate)` 与首次**逐字相同**（去掉 timestamp 后）。

    这是 `model/replay.ts` 判重键能认出它的唯一依据，也是 `events.ts` 敢用
    「同一块 + attempts++」而不是「新开一块」的前提。任何一项（step/node/message/detail）
    变了，判重就会静默失效 —— 表现是时间线上凭空多出一个块。
    """
    with _flow_client(tmp_path) as client:
        events = _run_approval_flow(client)

    dup = [
        e["event"]
        for e in events
        if e["event"]["type"] == "ToolCallStarted" and e["event"]["message"] == "delegate"
    ]
    assert len(dup) == 2

    def key(e: dict) -> str:
        return json.dumps(
            {k: v for k, v in e.items() if k != "timestamp"}, sort_keys=True, ensure_ascii=False
        )

    assert key(dup[0]) == key(dup[1]), (
        "重跑的开块与首次不再逐字相同，判重键会失效：\n"
        f"  首次: {key(dup[0])}\n  重跑: {key(dup[1])}"
    )


def test_pending_approval_has_no_delegate_completion(tmp_path):
    """**挂起时 delegate 没有 `ToolCallCompleted`** —— UI 判定「块还开着」的唯一依据。

    这条钉死 interrupt 的位置：它在子 agent 启动**之前**抛出、整个 tools 节点中止，
    所以收口事件根本没机会发。「挂起的块」本身就是待批准的可视化。
    """
    with _flow_client(tmp_path) as client:
        thread_id = _create(client)
        _start(client, thread_id, TASK)
        assert _wait_until(
            lambda: client.get(f"/sessions/{thread_id}").json()["status"] == "awaiting_approval"
        )
        rows = _raw(client, thread_id)

        assert any(r[1] == "ToolCallStarted" and r[3] == "delegate" for r in rows), (
            "挂起前应当已经发出 delegate 开块"
        )
        assert not any(
            r[1] == "ToolCallCompleted" and r[3].startswith("delegate(") for r in rows
        ), "挂起状态下不该有 delegate 的收口事件"
        assert not any(r[1] == "AgentStarted" and r[2] == "coder" for r in rows), (
            "挂起状态下子 agent 不该启动"
        )

        # 待批准 payload 的四个键
        approval = client.get(f"/sessions/{thread_id}").json()["approval"]
        assert len(approval) == 1
        assert set(approval[0]) == {"type", "specialist", "task", "question"}
        assert approval[0]["type"] == "approval"
        assert approval[0]["specialist"] == "coder"
        assert approval[0]["task"] == TASK


def test_rejection_closes_group_with_zero_child_runs(tmp_path):
    """拒绝批准：委派块以**零个子事件**闭合（`delegate()` 返回 ERROR 字符串而非抛异常）。

    → 折叠算法据此判定 `not_executed`。注意**无法区分**「被拒」与「specialist 名无法识别」
    （两条路径的返回值形状一致），所以 UI 文案写「未执行」而不是「被拒绝」。
    """
    with _flow_client(tmp_path) as client:
        thread_id = _create(client)
        _start(client, thread_id, TASK)
        assert _wait_until(
            lambda: client.get(f"/sessions/{thread_id}").json()["status"] == "awaiting_approval"
        )
        assert client.post(
            f"/sessions/{thread_id}/approval", json={"approved": False}
        ).status_code == 200
        assert _wait_until(
            lambda: client.get(f"/sessions/{thread_id}").json()["status"] == "idle"
        )
        rows = _raw(client, thread_id)

        assert not any(r[1] == "AgentStarted" and r[2] == "coder" for r in rows), "子 agent 不该启动"
        # 收口**照样发了** —— 且是 Completed 而非 Failed。这是最容易写错的判定：
        # 不能拿「delegate 完成了」反推子任务成功。
        assert any(
            r[1] == "ToolCallCompleted" and r[3].startswith("delegate(") for r in rows
        ), "拒绝后 delegate 仍应发出收口事件"
        assert not any(r[1] == "ToolCallFailed" for r in rows), (
            "拒绝不是工具失败 —— delegate 把 ERROR 当普通返回值交回去"
        )


def test_write_file_args_carry_full_content(tmp_path):
    """产出文件面板的全部依据：`write_file` 的 args 带 `path` + **完整 content**。

    `ToolCallStarted` 是唯一带 `detail={"args": …}` 的工具事件（Completed/Failed 都没有）。
    """
    with _flow_client(tmp_path) as client:
        events = _run_approval_flow(client)

    writes = [
        e["event"]
        for e in events
        if e["event"]["type"] == "ToolCallStarted" and e["event"]["message"] == "write_file"
    ]
    assert len(writes) == 1, f"期望 1 次 write_file，实际 {len(writes)}"
    args = writes[0]["detail"]["args"]
    assert args["path"] == "main.py"
    assert args["content"] == QUICKSORT_BODY, (
        "content 应当是整份文件原文，不是截断片段（`_brief_args` 只用在 message 里）"
    )

    # 反向钉死：Completed **不带** detail —— 所以 UI 拿不到工具结果原文，只能显示
    # 「工具名 + 参数」。这是后端限制，属于已知的待办项，不在这里绕过。
    completed = [
        e["event"]
        for e in events
        if e["event"]["type"] == "ToolCallCompleted"
        and e["event"]["message"].startswith("write_file(")
    ]
    assert completed, "没找到 write_file 的收口事件"
    assert completed[0]["detail"] is None, (
        "TOOL_CALL_COMPLETED 预期不带 detail —— UI 因此只能显示工具名 + 参数"
    )


def _settle(client: TestClient, thread_id: str) -> str:
    """等到会话落到一个稳定状态（`awaiting_approval` 或 `idle`）并返回它。

    ⚠️ 批准之后状态先变 `running`、再变回 `awaiting_approval` 或 `idle`，所以
    **不能**写 `while status == "awaiting_approval"` —— 那会在 `running` 那一刻
    误判成「全批完了」。
    """
    assert _wait_until(
        lambda: _status(client, thread_id) in ("awaiting_approval", "idle")
    ), f"会话没有落到稳定状态，当前 {_status(client, thread_id)}"
    return _status(client, thread_id)


class BatchRun(NamedTuple):
    """一次多委派会话的完整观测：事件流 + 终态 + 挂起轮数。"""

    events: list[dict]
    info: dict
    rounds: int


def _run_batch_flow(tmp_path, script: list, *, approvals_for=("coder",)) -> BatchRun:
    """跑一批多委派的会话，逐个批准直到 idle，返回事件流 + 终态。

    连**终态**一起返回是刻意的：多委派场景下事件流和会话状态会**互相矛盾**
    （见 `test_multiple_interrupts_in_one_batch_drop_later_delegates`），
    只钉一边就漏掉一半证据。
    """
    app = create_app(
        _settings(tmp_path),
        make_llm=lambda role: (
            _ScriptedLLM(script) if role == "Supervisor" else _TaskAwareLLM()
        ),
        require_approval_for=approvals_for,
    )
    with TestClient(app) as client:
        thread_id = _create(client)
        _start(client, thread_id, TASK)
        rounds = 0
        while _settle(client, thread_id) == "awaiting_approval":
            assert client.post(
                f"/sessions/{thread_id}/approval", json={"approved": True}
            ).status_code == 200
            rounds += 1
            assert rounds <= 5, "审批轮数失控"
        info = client.get(f"/sessions/{thread_id}").json()
        return BatchRun(_events(client, thread_id), info, rounds)


def _event_key(event: dict) -> str:
    """去掉 timestamp 的逐字比较键（判重键的可比形式）。"""
    return json.dumps(
        {k: v for k, v in event.items() if k != "timestamp"}, sort_keys=True, ensure_ascii=False
    )


def test_two_delegates_in_one_batch(tmp_path):
    """一批两个 `delegate`（只有第二个需批准）：重跑会把第一个委派**连同子事件整段重放**。

    这条钉死前端折叠算法必须做的两件事：

    1. 按**判重键**复用委派块 —— 不能只跟「当前打开的块」比。重放第一个委派的开块
       事件时，当前块早就是第二个的了，只比当前块就会把第一个委派重复开成第二张卡。
    2. 复用（重开）时**清空块内容**重新填 —— 否则第一张卡里会出现两份一模一样的
       write_file 行（第一轮的 + 重放的那一份）。

    没有这条契约，`model/events.ts` 的 `reopen` 分支就只能靠推测写。
    """
    events = _run_batch_flow(tmp_path, BATCH_TESTER_THEN_CODER).events
    rows = [(e["seq"], e["event"]["type"], e["event"]["agent"], e["event"]["message"]) for e in events]

    # 两个委派各自被重跑过 → 开块事件各出现 2 次（共 4 条）
    opens = [r for r in rows if r[1] == "ToolCallStarted" and r[3] == "delegate"]
    assert len(opens) == 4, f"期望 4 条开块事件（两个委派各重放一次），实际 {len(opens)}"

    by_seq = {e["seq"]: e["event"] for e in events}
    # a.py（tester，第一个）的两条开块事件逐字相同；b.py（coder）同理
    assert _event_key(by_seq[opens[0][0]]) == _event_key(by_seq[opens[2][0]]), (
        "第一个委派的两条开块事件应当逐字相同"
    )
    assert _event_key(by_seq[opens[1][0]]) == _event_key(by_seq[opens[3][0]]), (
        "第二个委派的两条开块事件应当逐字相同"
    )
    # 但两个委派之间必须**不同** —— 否则会被误并成一张卡
    assert _event_key(by_seq[opens[0][0]]) != _event_key(by_seq[opens[1][0]]), (
        "两个委派的开块事件必须有不同判重键（specialist/task 不同）"
    )
    # 顺序：第一个委派的开块（+重放）都在第二个之前
    assert opens[0][0] < opens[1][0] < opens[2][0] < opens[3][0], (
        "重放顺序应当是 S(a) S(b) S(a) S(b) —— 第二个委派是在第二轮才被 interrupt 打断的"
    )

    # 子事件确实被整段重放过：tester 跑 2 次、coder 跑 1 次（它在第一轮根本没轮到）
    assert len([r for r in rows if r[1] == "AgentStarted" and r[2] == "tester"]) == 2
    assert len([r for r in rows if r[1] == "AgentStarted" and r[2] == "coder"]) == 1
    writes = [r for r in rows if r[1] == "ToolCallStarted" and r[3] == "write_file"]
    assert len(writes) == 3, f"tester 跑 2 次 + coder 跑 1 次 = 3 次 write_file，实际 {len(writes)}"

    # ⚠️ **开块 4 条、收口只有 3 条** —— 这个不对称是整套协议的关键形状，不是数据缺失：
    # seq 10 的 S(b) 发出之后，`interrupt()` 在 delegate(coder) 的执行体里抛出
    # （`supervisor.py:84`），tools 节点的 `for` 循环**就地中断** —— 那一轮的
    # `C(b)` 永远发不出来。于是恒等式是「开块数 = 收口数 + 1」，多出来的那个正是
    # 被 interrupt 打断的委派，而它**恰好是待批准状态在图上的唯一可视证据**
    # （`interrupts` 非空 ⟺ 有一个开着的块）。
    #
    # 前端折叠算法必须容忍这个悬空的开块：它在第二轮被**逐字重发**时命中判重键、
    # 变成同一个块的 `attempts++`，然后在 seq 26 收到自己那条迟到的 C(b)。
    closes = [r for r in rows if r[1] == "ToolCallCompleted" and r[3].startswith("delegate(")]
    assert len(closes) == 3, f"期望 3 条收口事件（悬空的那个是被打断的 coder），实际 {len(closes)}"
    assert [r[3] for r in closes] == [
        "delegate(specialist='tester', task='写 a.py')",
        "delegate(specialist='tester', task='写 a.py')",
        "delegate(specialist='coder', task='写 b.py')",
    ], "tester 收口两次（两轮各一次），coder 只在第二轮收口"

    # 按轮次切开看更清楚（第二轮从第一个委派的**重发**开始）：
    #   第一轮（seq < opens[2]）：开块 {a, b}，收口 {a}       → b 悬空
    #   第二轮（seq >= opens[2]）：开块 {a, b}，收口 {a, b}   → 平衡
    round2 = opens[2][0]
    assert [r[0] for r in opens if r[0] < round2] == [opens[0][0], opens[1][0]]
    assert [r[0] for r in closes if r[0] < round2] == [closes[0][0]], (
        "第一轮只有第一个委派收了口 —— 第二个委派的 C 被 interrupt 吃掉了"
    )
    assert [r[0] for r in opens if r[0] >= round2] == [opens[2][0], opens[3][0]]
    assert [r[0] for r in closes if r[0] >= round2] == [closes[1][0], closes[2][0]], (
        "第二轮两个委派都收了口（重发的那条开块事件最终等到了自己迟到的收口）"
    )


def test_multiple_interrupts_in_one_batch_drop_later_delegates(tmp_path):
    """⚠️ **后端缺陷的特征化测试**（P7 遗留；P8 的后端零改动约束下只记录、不修）。

    一批里有两个**都需要批准**的委派时，一次 tools 节点执行里会出现 2 个 `interrupt()`
    （`supervisor.py:84`，每个需批准的 delegate 各一个）。实测：**只有第一个能拿到
    `Command(resume="yes")`，第二个被静默吞掉** —— 图不再挂起、也不报错，直接从
    `run_task` 的 `while` 里以「`snap.next` 为空」退出，而 state 里的 `status` 还停在
    `"running"`。`runtime/session.py:459-467` 于是补发一条**图外**的
    `AgentFailed(message="状态 running")`，会话落到 `idle`。

    与「`snap.interrupts` 会一轮一轮逐个冒出来」的朴素预期**矛盾**：实际情况是**第二轮
    重跑到第二个 interrupt 就整个咽掉了**（没有第二次挂起）。

    用探针收敛出的判据（三个变体，`require_approval_for` 与脚本不同）：

    | 一批里的委派 | 一次节点执行里的 interrupt 数 | 结果 |
    |---|---|---|
    | coder + coder | 2 | ❌ 第二个被吞，`result.status=running` |
    | coder + tester | 2 | ❌ 同上 |
    | tester + coder | 1 | ✅ 正常跑完，`result.status=finished` |

    → 判据是「**同一次节点执行里的 interrupt 个数 > 1**」，与 specialist 是否同名无关
    （同名只是让第二个也进需批准集合，不必需）。

    **对前端的后果（这才是这条用例挂在契约测试里的理由）**：事件流会以
    「一个开着、永远收不了口的委派块 + 一条根 `AgentFailed`」收尾，末尾**没有任何
    `ERROR:` 文本回流、`info.error` 也仍是 `None`**。所以 UI 不能把「流结束」当成
    「任务成功」，必须同时看 `result.status` / 末尾的 `AgentFailed`。

    修法（不属于 P8，因为要动 `supervisor.py` / `driver.py`）：给 `interrupt()` 传显式
    `id=`，用 `Command(resume={id: value})` 按 id 恢复；再给 `driver.run_task` 加一道
    断言 —— 「`snap.next` 空、但 state 里的 `status` 还是 `running`」正是 interrupt
    被吞的签名，应当报错而不是当成正常收尾。

    这条用例**故意钉住错误行为**：后端修好后它会红，那时连同这段 docstring 一起更新。
    """
    run = _run_batch_flow(tmp_path, BATCH_CODER_TWICE)
    rows = [
        (e["seq"], e["event"]["type"], e["event"]["agent"], e["event"]["message"])
        for e in run.events
    ]

    # 开块 3 条：a.py 两次（首轮 + 重跑后的逐字重发），b.py **一次**
    # —— b 的开块事件是发出来了的（说明节点确实走到了第二个 tool_call），
    #   吞掉的是它后面那个 interrupt，不是 STARTED 这个 emit。
    opens = [r for r in rows if r[1] == "ToolCallStarted" and r[3] == "delegate"]
    assert len(opens) == 3, f"缺陷场景下开块事件应当是 3 条，实际 {len(opens)}"
    by_seq = {e["seq"]: e["event"] for e in run.events}
    assert _event_key(by_seq[opens[0][0]]) == _event_key(by_seq[opens[1][0]]), (
        "前两条是 a.py 的首轮与重发（逐字相同），正是「答一个 → 节点重跑」的痕迹"
    )
    assert "b.py" in json.dumps(by_seq[opens[2][0]]["detail"], ensure_ascii=False)

    # 只有第一个委派收了口；第二个**永远收不了口**
    closes = [r for r in rows if r[1] == "ToolCallCompleted" and r[3].startswith("delegate(")]
    assert len(closes) == 1 and "a.py" in closes[0][3], "只有 a.py 收了口 —— b.py 被丢弃"
    assert run.rounds == 1, f"只该挂起一次（第二个 interrupt 被吞），实际 {run.rounds} 次"

    # b.py 的子 agent 从未启动、文件从未落盘
    assert len([r for r in rows if r[1] == "AgentStarted" and r[2] == "coder"]) == 1
    writes = [r[3] for r in rows if r[1] == "ToolCallCompleted" and r[3].startswith("write_file")]
    assert len(writes) == 1 and "a.py" in writes[0], "只有 a.py 被写出来"
    assert sorted(p.name for p in (tmp_path / "ws").rglob("*.py")) == ["a.py"], (
        "b.py 从未落盘 —— 这是「静默丢弃」最硬的证据（不是渲染问题，是文件真的没有）"
    )

    # 收尾是**图外**那条 AGENT_FAILED（step/node 都是 None），message 就是「状态 running」
    failures = [r for r in rows if r[1] == "AgentFailed"]
    assert len(failures) == 1
    assert failures[0][2] == "Supervisor"
    assert failures[0][3] == "状态 running"
    assert by_seq[failures[0][0]]["step"] is None and by_seq[failures[0][0]]["node"] is None

    # 会话终态：**`idle` 但 result.status 仍停在 `running`** —— 这个自相矛盾正是
    # 缺陷的标志，也是 UI 不能只看 `status === "idle"` 就宣布成功的理由。
    assert run.info["status"] == "idle"
    assert run.info["result"]["status"] == "running"
    assert run.info["result"]["result"] is None
    assert run.info["error"] is None, "会话级 error 是 None —— 没有任何地方把它当失败"


def test_batch_fixture_is_written(tmp_path):
    """把批处理夹具写出来（**正常跑完**的那个形状），给前端折叠测试用。"""
    run = _run_batch_flow(tmp_path, BATCH_TESTER_THEN_CODER)
    _write_fixture(BATCH_FIXTURE, _normalize(run.events))
    saved = json.loads(BATCH_FIXTURE.read_text(encoding="utf-8"))["events"]
    assert len(saved) == len(run.events)
    assert any(
        e["event"]["type"] == "ToolCallStarted" and e["event"]["message"] == "write_file"
        for e in saved
    )


def test_real_stream_fixtures_are_written(tmp_path):
    """跑两条真实会话（批准 / 拒绝）→ 写成前端夹具（归一化 thread_id / timestamp）。

    两个夹具覆盖折叠算法的两条主分支：`completed`（含挂起重跑）与 `not_executed`。
    都对着**后端真正发出来的字节**验证，而不是对着谁推演的序列。

    两条流程各用一个 app：脚本 LLM 是同一条 `_ScriptedLLM`（`pop` 掉就没了），
    同一进程里跑第二遍会拿到 `(fallback done)`。
    """
    with _flow_client(tmp_path) as client:
        approved = _run_approval_flow(client)
    with _flow_client(tmp_path) as client:
        rejected = _run_approval_flow(client, approve=False)

    _write_fixture(APPROVED_FIXTURE, _normalize(approved))
    _write_fixture(REJECTED_FIXTURE, _normalize(rejected))

    for path in (APPROVED_FIXTURE, REJECTED_FIXTURE):
        assert path.is_file(), f"{path.name} 没写出来"
        saved = json.loads(path.read_text(encoding="utf-8"))["events"]
        # 夹具里不该残留真实的 thread_id
        assert all(e["event"]["thread_id"] == "<thread>" for e in saved)
        # 判重键依赖的字段必须原样保留
        for e in saved:
            assert isinstance(e["event"]["step"], int) or e["event"]["step"] is None
            assert isinstance(e["event"]["node"], str) or e["event"]["node"] is None
            assert isinstance(e["event"]["message"], str)

    # 两条流程的形状差异必须真的落在夹具里（否则前端测试会静默测同一个东西）
    assert any(
        e["event"]["type"] == "ToolCallStarted" and e["event"]["message"] == "write_file"
        for e in json.loads(APPROVED_FIXTURE.read_text(encoding="utf-8"))["events"]
    ), "批准夹具里应当有 write_file"
    assert not any(
        e["event"]["type"] == "ToolCallStarted" and e["event"]["message"] == "write_file"
        for e in json.loads(REJECTED_FIXTURE.read_text(encoding="utf-8"))["events"]
    ), "拒绝夹具里不该有 write_file"
    assert len(approved) != len(rejected), "两条流程的事件条数不该相同"


# ---------------------------------------------------------------- 3. 错误形状


def test_error_shapes_match_client_ts(tmp_path):
    """404（字符串）/ 422（数组）/ 409（对象）三种 detail —— `client.ts` 的归一化分支。"""
    with _client(tmp_path) as client:
        thread_id = _create(client)

        not_found = client.get("/sessions/nope")
        assert not_found.status_code == 404
        assert isinstance(not_found.json()["detail"], str)

        invalid = client.post(f"/sessions/{thread_id}/messages", json={})
        assert invalid.status_code == 422
        assert isinstance(invalid.json()["detail"], list)
        assert invalid.json()["detail"], "校验错误数组不该是空的"


def test_busy_409_detail_is_object(tmp_path):
    """409（忙）的 detail 是**对象** `{status, message}` —— 与 404 的字符串形状不同。

    这是 `client.ts` 必须分两条分支的原因：前端要拿 `detail.status` 区分
    「正在跑（转圈）」与「等审批（弹按钮）」。
    """
    gate = threading.Event()
    with _client(tmp_path, gate=gate) as client:
        thread_id = _create(client)
        _start(client, thread_id)
        assert _wait_until(lambda: client.get(f"/sessions/{thread_id}").json()["status"] == "running")

        busy = client.post(f"/sessions/{thread_id}/messages", json={"task": "再来一个"})
        assert busy.status_code == 409
        detail = busy.json()["detail"]
        assert isinstance(detail, dict), f"409 忙的 detail 应当是对象，实际 {type(detail)}"
        assert set(detail) == {"status", "message"}
        assert detail["status"] == "running"

        # ⚠️ `gate.set()` 只是**放行**，不是**等它跑完** —— 必须再等会话落到终态才能
        # 退出 `with`。否则 app 拆除（关掉 SqliteSaver 的连接）会撞上仍在跑的 worker
        # 线程里那次还没落地的 checkpoint `put`：sqlite3 对象已被释放，
        # 报 **`Windows fatal exception: access violation`** —— C 层崩溃，
        # `try/except` 抓不住，整个 pytest 进程原地死掉（实测 8 次里崩 5 次，
        # 且崩在这一条上却显示前一条的名字，极难定位）。
        # `test_api.py:292-293` 的 P7 原版就是这么写的，这里照抄。
        gate.set()
        assert _wait_until(lambda: _status(client, thread_id) == "idle")


# ---------------------------------------------------------------- 4. 构建产物托管


BUILT_INDEX = Path(__file__).resolve().parent.parent / "api" / "static" / "index.html"

# 「是 Vite 产物」而不只是「index.html 存在」：P7 那个单文件测试页也叫 index.html，
# 它在 P8 里被删掉了，但本地可能还躺着没清的历史文件 —— 只判 `is_file()` 会让这条
# 用例对着一个 vanilla 页面假绿。`assets/` 引用是 Vite 产物的签名。
BUILT_FROM_VITE = (
    BUILT_INDEX.is_file() and "assets/" in BUILT_INDEX.read_text(encoding="utf-8", errors="ignore")
)

needs_build = pytest.mark.skipif(
    not BUILT_FROM_VITE,
    reason="api/static 下没有 Vite 构建产物（产物已 gitignore，先跑 npm run build）",
)


@needs_build
def test_serves_built_ui(tmp_path):
    """构建产物取得到，且**缓存策略按「文件名会不会随内容变」分两半**。

    缓存断言是 2026-09-25 补的，起因是用户实测反馈「前端改了、界面还是老样子」：
    入口 HTML 就是一份资源清单（引用带内容哈希的文件名），重建后哈希会变，而 Starlette
    默认只发 `etag`/`last-modified` → 浏览器启发式复用、不回源校验 → 拿旧清单配旧文件，
    整套界面退回改动前（实测症状：委派卡压成 2px + 时间线滚不动）。策略见
    `api/app.py` 的 `_HashedAssetStaticFiles`。

    当晚又从 `no-cache` 改成 `no-store` + **摘掉验证器**（用户反馈「每次一进去都要刷新」）：
    `no-cache` 只在「浏览器会把 304 里的 `cache-control` 合并回已存副本」这条**无法验证的
    依赖**成立时才管用，而旧副本的响应头是和副本一起存下来的、服务端再也够不着。所以这里
    断言的是**强性质**：入口 HTML 既不许被存，也**不许出现 304**（`If-None-Match: *` 是
    条件请求里最强的形式，会走 304 的就必须答 304）—— 没有可存的副本，就没有「旧副本」这个状态。
    """
    with _client(tmp_path) as client:
        root = client.get("/")
        assert root.status_code == 200, root.text
        assert '<div id="root">' in root.text, "返回的不是 Vite 构建出的 index.html"
        # 入口 HTML 既不许被缓存，也不许有条件请求这条路可走
        assert root.headers["cache-control"] == "no-store", "入口 HTML 必须禁止存储"
        assert "etag" not in root.headers, "入口 HTML 不能发 etag（否则又能走 304）"
        assert "last-modified" not in root.headers, "入口 HTML 不能发 last-modified"
        # 条件请求必须**答 200 整份**，而不是 304 —— 这是「摘掉验证器」的核心性质
        conditional = client.get("/", headers={"If-None-Match": "*"})
        assert conditional.status_code == 200, (
            f"入口 HTML 对条件请求答了 {conditional.status_code}，"
            "说明又走上了 304 那条路（旧副本会因此永远活着）"
        )
        assert conditional.headers["cache-control"] == "no-store"

        # ⛔ 上面那三条只管得住**将来**存的副本；**已经存在浏览器里**的那份，
        # `no-store` 追不到它（缓存键是完整 URL，而选中会话是 `history.replaceState`、
        # 不产生导航 → 那份旧副本挂在 `/` 上，刷新却发生在 `/?session=…`）。
        # 唯一还能碰到它的手段是在交付文档时叫浏览器清缓存 → 清一次永久生效。
        assert root.headers["clear-site-data"] == '"cache"', (
            "入口 HTML 必须带 Clear-Site-Data: \"cache\" —— 否则浏览器里已存的旧副本"
            "永远清不掉（用户那句「每次一进去都要刷新」就是这么来的）"
        )

        # 页面引用的 hashed asset 必须真的能取到，且可以长缓存（名字随内容变，老名字不会再来）
        refs = re.findall(r'(?:src|href)="(/assets/[^"]+)"', root.text)
        assert refs, "index.html 里没有 /assets/* 引用"
        for ref in refs:
            got = client.get(ref)
            assert got.status_code == 200, f"{ref} 取不到"
            assert "immutable" in got.headers["cache-control"], f"{ref} 没拿到长缓存"
            # **产物绝不能带 Clear-Site-Data**：那会让每次资源加载都清一次缓存，病态。
            assert "clear-site-data" not in got.headers, (
                f"{ref} 带了 clear-site-data —— 只能发给文档，不能发给产物"
            )
