"""P9: 会话与事件的「读路径」—— 重启后从持久化重建列表与历史。

**「重启」在这里就是同一个 `tmp_path` 上先后起两个 `create_app`**：checkpoint 库与
workspace 目录都在，进程内存全清（registry 每次 lifespan 重建）。这正是用户实测的
那个场景（重启 uvicorn → 列表空、历史全无，而库里明明有），只是把真 uvicorn 换成了
TestClient —— 差异只在 ASGI 传输层，读路径的每一行代码都一样。

钉住的是六条范围决策的落点：
- ① 全部列出、最近活动在前
- ② 非正常收尾 → `interrupted` 只读（**端到端未自动化，见下面 `_derive` 一节的理由**）
- ③ 能看 + 能续跑（追加指令真的落到同一个 workspace）
- ④ sqlite → 列空壳会话 + `history_available=false`
- ⑤ DELETE 连库一起删（重启不复现）
- ⑥ 停在「待批准」的会话重启后仍能点批准（**跨进程 resume 已由真机 spike 实证**，
  这里钉的是服务端这条 HTTP 路径）

脚手架（`_create`/`_start`/`_await_status`/…）直接复用 `test_api` 的：**必须**与 P7 的
服务端用例走同一套装配，否则「重启后仍然可用」会被我自己另写的一套接线掩盖掉。
"""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from api.app import create_app
from config.settings import Settings
from persistence.checkpointer import list_checkpoint_threads
from runtime.catalog import derive_status, record_from_snapshot
from runtime.registry import SessionRegistry
from runtime.session import RestorePayload, Session, SessionStatus
from test_api import (  # noqa: F401  同目录测试模块，复用 P7 的服务端脚手架
    QUICKSORT,
    _await_events_settled,
    _await_status,
    _create,
    _start,
    _status,
    _tool_call,
    _wait_until,
)
from test_api import _ScriptedLLM

# ---------------------------------------------------------------- 脚本

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

# 第二轮：**故意写另一个文件**——这样「续跑」的断言落在文件系统上，而不是落在
# 某个可能自己就变了的计数器上。
SECOND_ROUND = {
    "Supervisor": [
        _tool_call(1, "delegate", {"specialist": "coder", "task": "再写一个 util.py"}),
        AIMessage(content="完成：util.py 已写好。"),
    ],
    "coder": [
        _tool_call(1, "write_file", {"path": "util.py", "content": "X = 1\n"}),
        AIMessage(content="已写 util.py"),
    ],
}

# 停在待批准之后重启：Supervisor 只剩收尾这一步，coder 子图是**重新 invoke**
# （新 delegate-N thread），所以它的脚本要重新给全。
RESUME_TAIL = {
    "Supervisor": [AIMessage(content="完成：Coder 已写好快排。")],
    "coder": DELEGATE_TO_CODER["coder"],
}


# ---------------------------------------------------------------- 脚手架


def _llm_factory(scripts=None):
    scripts = scripts if scripts is not None else DELEGATE_TO_CODER

    def make_llm(role):
        return _ScriptedLLM(scripts.get(role, []))

    return make_llm


def _sqlite_settings(tmp_path) -> Settings:
    return Settings(
        llm_api_key="fake-key",
        llm_base_url="",
        llm_model="fake",
        workspace_root=Path(tmp_path) / "ws",
        checkpoint_db_path=Path(tmp_path) / "data" / "cp.db",
        session_idle_timeout=1800.0,
    )


@contextmanager
def _client(settings: Settings, *, scripts=None, approvals=()):
    """起一个真 app（真 lifespan）+ TestClient。

    与 `test_api._client` 的唯一差别是**接 settings 对象本身**（那个只接 tmp_path）——
    P9 的用例要能用同一个 Settings 起两次 app（这就是「重启」），还要能切成 PG。
    """
    app = create_app(
        settings,
        make_llm=_llm_factory(scripts),
        require_approval_for=approvals,
    )
    with TestClient(app) as client:
        yield client


def _list(client) -> dict:
    response = client.get("/sessions")
    assert response.status_code == 200, response.text
    return response.json()


def _ids(client) -> list[str]:
    return [s["thread_id"] for s in _list(client)["sessions"]]


def _info(client, thread_id: str) -> dict:
    response = client.get(f"/sessions/{thread_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _events(client, thread_id: str, **params) -> list[dict]:
    response = client.get(f"/sessions/{thread_id}/events", params=params)
    assert response.status_code == 200, response.text
    return response.json()["events"]


def _workspace(tmp_path, thread_id: str, name: str) -> Path:
    return Path(tmp_path) / "ws" / thread_id / name


def _await_new_events_settled(client, thread_id: str, before: int, timeout: float = 30.0) -> None:
    """等到**新增**事件出现并收尾，条数必须超过 `before`。

    ⚠️ 恢复出来的会话不能用 `test_api._await_events_settled`：它的判据是「末条是
    AgentCompleted/AgentFailed」，而**旧事件流的末条通常正是它** → 等待立刻返回，
    用例便在「新任务刚起跑」时去读产物，拆 app 时又会把正在跑的 worker 从底下抽走
    （实测到过：事件 store 的池已关、连续写失败 3 次后闩锁）。加上条数条件才是
    「这一轮真的跑完了」。
    """
    def settled() -> bool:
        events = _events(client, thread_id)
        return (
            len(events) > before
            and events[-1]["event"]["type"] in ("AgentCompleted", "AgentFailed")
        )

    assert _wait_until(settled, timeout), f"新增事件没有收尾（起始 {before} 条）"


def _checkpoint_threads(settings: Settings) -> list[str]:
    """直接查 sqlite 库里的 thread 列表（绕开服务，验证「真的从库里删掉了」）。"""
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as saver:
        return list_checkpoint_threads(saver)


# ---------------------------------------------------------------- ① 列表 / ③ 续跑（sqlite）


def test_restart_lists_finished_session_and_resumes_it(tmp_path):
    """**用户报的那个问题的正身**：跑完一个会话 → 重启 → 它还在列表里，而且能续跑。

    续跑的证据落在文件系统上：第二轮写的是**另一个文件**（`util.py`），
    所以「图真的接着旧历史跑下去了」不是靠某个计数器推出来的。
    """
    settings = _sqlite_settings(tmp_path)
    with _client(settings, scripts=DELEGATE_TO_CODER) as client:
        _create(client, "keep")
        _start(client, "keep")
        _await_events_settled(client, "keep")
        assert _workspace(tmp_path, "keep", "main.py").read_text(encoding="utf-8") == QUICKSORT

    # ------------------------------ 重启（内存全清、库与 workspace 都在）
    with _client(settings, scripts=SECOND_ROUND) as client:
        assert _ids(client) == ["keep"], "重启后列表里必须还有这个会话"
        assert _info(client, "keep")["status"] == "idle"  # finished → 可续跑

        _start(client, "keep", "再写一个 util.py")
        _await_new_events_settled(client, "keep", before=0)

        assert _workspace(tmp_path, "keep", "util.py").read_text(encoding="utf-8") == "X = 1\n"
        assert _workspace(tmp_path, "keep", "main.py").exists(), "续跑不该清掉旧产物"


def test_restart_lists_most_recent_first(tmp_path):
    """① 最近活动在前。三个会话按 a→b→c 依次跑，重启后必须是 c, b, a。"""
    settings = _sqlite_settings(tmp_path)
    with _client(settings, scripts=DELEGATE_TO_CODER) as client:
        for name in ("aaa", "bbb", "ccc"):
            _create(client, name)
            _start(client, name)
            _await_events_settled(client, name)

    with _client(settings) as client:
        assert _ids(client) == ["ccc", "bbb", "aaa"]


def test_restart_does_not_materialize_everything_up_front(tmp_path):
    """历史会话是**懒物化**的：列表不该顺手把每个会话的图都建起来。

    判据用注册表的内存视图（`sessions()` 只含已物化的）：列完之后一个都不该在里面，
    点开一个才出现一个。这条挡的是「启动时 N×60ms + N 个空 workspace 目录」
    那个退化实现 —— 更要命的是那些被物化的会话会被回收线程收走，
    历史于是**第二次**从列表里消失。
    """
    settings = _sqlite_settings(tmp_path)
    with _client(settings, scripts=DELEGATE_TO_CODER) as client:
        for name in ("one", "two"):
            _create(client, name)
            _start(client, name)
            _await_events_settled(client, name)

    with _client(settings) as client:
        assert len(_ids(client)) == 2
        registry = client.app.state.registry
        assert registry.sessions() == [], "列一下列表就物化了，懒物化没生效"

        _info(client, "one")  # 点开一个
        assert [s.thread_id for s in registry.sessions()] == ["one"]


def test_create_on_a_restored_id_is_409(tmp_path):
    """`POST /sessions` 撞上一个**只存在于持久化里**的 id 必须 409。

    放过它的话会造出一个 `_events` 为空的新会话压在同一个 thread 上：图形 state
    是真的（续跑会接上旧历史），但用户看不到刚续上的那段历史 —— 比报错糟得多。
    """
    settings = _sqlite_settings(tmp_path)
    with _client(settings, scripts=DELEGATE_TO_CODER) as client:
        _create(client, "taken")
        _start(client, "taken")
        _await_events_settled(client, "taken")

    with _client(settings) as client:
        assert client.post("/sessions", json={"thread_id": "taken"}).status_code == 409
        # 原会话不被动过
        assert _info(client, "taken")["status"] == "idle"


# ---------------------------------------------------------------- ⑥ 待批准（sqlite）


def test_restart_keeps_awaiting_approval_and_can_still_approve(tmp_path):
    """⑥ 停在「待批准」的会话，重启后仍标「待批准」，**点批准真的把活干完**。

    跨进程 `Command(resume="yes")` 已在真机 spike 上实证（PG 与 sqlite 各一次：
    全新 Python 进程里恢复挂起会话 → 批准 → `main.py` 真落盘、seq 从 0 连续）。
    这条用例钉的是服务端把这条路走通：HTTP 端点 → 懒物化 → `Session.resume`。
    """
    settings = _sqlite_settings(tmp_path)
    with _client(settings, scripts=DELEGATE_TO_CODER, approvals=("coder",)) as client:
        _create(client, "wait")
        _start(client, "wait")
        _await_status(client, "wait", "awaiting_approval")
        assert _info(client, "wait")["approval"], "挂起时必须有待批准内容"
        assert not _workspace(tmp_path, "wait", "main.py").exists(), "没批准就不该动文件"

    # ------------------------------ 重启
    with _client(settings, scripts=RESUME_TAIL, approvals=("coder",)) as client:
        info = _info(client, "wait")
        assert info["status"] == "awaiting_approval"
        assert info["approval"][0]["specialist"] == "coder"

        response = client.post("/sessions/wait/approval", json={"approved": True})
        assert response.status_code == 200, response.text
        _await_status(client, "wait", "idle")

        assert _workspace(tmp_path, "wait", "main.py").read_text(encoding="utf-8") == QUICKSORT


# ---------------------------------------------------------------- ⑤ DELETE 连库一起删


def test_delete_then_restart_does_not_resurrect(tmp_path):
    """⑤ DELETE 必须是**真删**（checkpoint 也删），否则「删了→重启→又回来」。

    两处断言：REST 层重启后 404，以及**直接打开 sqlite 库**看 thread 是否真没了
    （只看 REST 的话，一个「删了内存、库还在」的实现也能骗过这条用例 —— 直到重启）。
    """
    settings = _sqlite_settings(tmp_path)
    with _client(settings, scripts=DELEGATE_TO_CODER) as client:
        _create(client, "gone")
        _start(client, "gone")
        _await_events_settled(client, "gone")
        assert client.delete("/sessions/gone").status_code == 204

    assert "gone" not in _checkpoint_threads(settings), "checkpoint 没被删掉"

    with _client(settings) as client:
        assert _ids(client) == []
        assert client.get("/sessions/gone").status_code == 404


def test_delete_restored_session_without_materializing_it(tmp_path):
    """删一个**只在持久化里**的会话（从没被点开过）也要连库删干净。

    这条路径走的是 `_known` 分支：会话没被物化，所以「关资源」那一步是空的，
    全靠 `catalog.purge()` 把库删掉 —— 漏了这一半就是「删了重启又回来」。
    """
    settings = _sqlite_settings(tmp_path)
    with _client(settings, scripts=DELEGATE_TO_CODER) as client:
        _create(client, "ghost")
        _start(client, "ghost")
        _await_events_settled(client, "ghost")

    with _client(settings) as client:
        registry = client.app.state.registry
        assert registry.sessions() == []  # 确认没被物化过
        assert client.delete("/sessions/ghost").status_code == 204

    assert "ghost" not in _checkpoint_threads(settings)

    with _client(settings) as client:
        assert _ids(client) == []


# ---------------------------------------------------------------- ④ sqlite 的能力降级


def test_sqlite_degrades_to_shell_sessions_with_explicit_flag(tmp_path):
    """④ sqlite 下会话能列出来但**事件流是真的没有**，能力位必须如实说。

    「点进去时间线是空的」有两种可能：会话本来就没内容，或者**事件从来没落过库**。
    用户没法从界面上分辨这两者，所以必须由 `history_available` 说清楚 ——
    这也是「不假装 sqlite 有历史」的唯一落点。
    """
    settings = _sqlite_settings(tmp_path)
    with _client(settings, scripts=DELEGATE_TO_CODER) as client:
        _create(client, "shell")
        _start(client, "shell")
        _await_events_settled(client, "shell")
        live_events = _events(client, "shell")
        assert len(live_events) >= 5, "重启前事件是在内存里的"
        live_result = _info(client, "shell")["result"]

    with _client(settings) as client:
        body = _list(client)
        assert body["history_available"] is False
        info = body["sessions"][0]
        assert info["thread_id"] == "shell"
        assert info["event_count"] == 0, "sqlite 下事件从不落库"
        assert _events(client, "shell") == []  # 空时间线，不是 404

        # 结果是**从 checkpoint 的 state 重建**的，所以「能看结果、看不到过程」
        assert info["result"] == live_result


# ---------------------------------------------------------------- ② 状态推导（纯函数 + 会话行为）

# ② 的**端到端**重启未自动化，理由记在这里而不是留白：造一个「进程被杀时正在跑」的
# 现场，需要让一个 worker 线程正卡在 `graph.invoke` 里，然后关掉它脚下的 sqlite saver
# —— 那正是实测到过一次访问违例（langgraph checkpoint put 线程与 close 竞争）的配方。
# 与其在测试套件里复现一个已知会崩的场景，不如把判据用纯函数钉死（下面两条），
# 真正的跨进程现场由真机 spike 覆盖（进程整个退出，不存在这个竞争）。


def test_derive_status_table():
    """状态推导表：**判序**是这张表的全部难度所在。"""
    assert derive_status({}, has_next=False, has_interrupts=False) is SessionStatus.IDLE
    assert derive_status({"status": "finished"}, has_next=False, has_interrupts=False) is SessionStatus.IDLE
    # `error` 也是收尾态 → 可以续跑（③）
    assert derive_status({"status": "error"}, has_next=False, has_interrupts=False) is SessionStatus.IDLE

    # 有未执行节点 / state 还写着 running → 只读
    assert derive_status({"status": "running"}, has_next=True, has_interrupts=False) is SessionStatus.INTERRUPTED
    assert derive_status({"status": "running"}, has_next=False, has_interrupts=False) is SessionStatus.INTERRUPTED

    # ⚠️ 停在审批点的会话同时满足「有未执行节点」和「state 写着 running」
    #（interrupt 就在 tools 节点里抛出）—— 所以 interrupts 必须**先判**，
    # 否则「停在审批点」会被静默降级成只读。
    assert derive_status({"status": "running"}, has_next=True, has_interrupts=True) is SessionStatus.AWAITING_APPROVAL


def test_record_from_snapshot_takes_approval_payload_from_interrupts():
    """`approval` 的取值表达式必须与 `driver.run_task` 逐字一致，否则 UI 拿到的形状会变。"""
    snap = SimpleNamespace(
        values={"status": "running", "iteration_count": 3},
        next=("tools",),
        interrupts=[SimpleNamespace(value={"type": "approval", "specialist": "coder"})],
    )
    record = record_from_snapshot("t1", snap, 7, last_id=42)

    assert record.status is SessionStatus.AWAITING_APPROVAL
    assert record.approval == ({"type": "approval", "specialist": "coder"},)
    assert record.event_count == 7
    assert record.last_id == 42
    assert record.snapshot()["status"] == "awaiting_approval"


def test_interrupted_session_is_readonly_but_reapable(tmp_path):
    """② `interrupted` 会话：**只读**，但必须**可回收**。

    可回收这条是防泄漏的硬要求：每点开一个历史会话就物化一个 graph + saver，
    如果它既不能跑（`begin()` 只认 IDLE）又进不了回收白名单，那就永久占着 ——
    点开 20 个历史会话就泄漏 20 份。
    """
    settings = _sqlite_settings(tmp_path)
    session = Session(
        "half",
        settings,
        workspace_root=Path(tmp_path) / "ws" / "half",
        make_llm=_llm_factory(),
        checkpointer=None,
        restore=RestorePayload(status=SessionStatus.INTERRUPTED),
    )
    try:
        assert session.status is SessionStatus.INTERRUPTED
        assert session.begin("再来一个") is False, "已中断的会话不能被当新任务起跑"
        assert session.resume(True) is False, "没在等审批就不能批准"
        assert session.reapable(0.0) is True
    finally:
        session.close()


@pytest.mark.parametrize(
    "restored,expected",
    [
        (SessionStatus.IDLE, True),
        (SessionStatus.AWAITING_APPROVAL, True),
        (SessionStatus.INTERRUPTED, True),  # P9 新增：不回收 = 每点开一个历史就泄漏一份
        (SessionStatus.RUNNING, False),  # 会在 LLM 调用进行中把容器 rm -f
        (SessionStatus.CLOSED, False),
    ],
)
def test_reapable_covers_every_supported_status(restored, expected, tmp_path):
    """回收白名单的表驱动断言。

    `RUNNING` 那一行是**故意构造**的：`derive_status` 永远不会产出 running（它把
    那种形状判成 interrupted），所以这个组合只能来自「有人写错了恢复逻辑」——
    而它的后果最严重（既不能 begin 也不能回收，只能 DELETE）。这里把它钉死。
    """
    settings = _sqlite_settings(tmp_path)
    session = Session(
        "s",
        settings,
        workspace_root=Path(tmp_path) / "ws" / "s",
        make_llm=_llm_factory(),
        restore=RestorePayload(status=restored),
    )
    try:
        assert session.reapable(0.0) is expected
    finally:
        session.close()


# ---------------------------------------------------------------- 回收 ≠ 历史消失


class _FakeCatalog:
    """最小目录替身：只记 discover/purge，用来验证注册表的协作侧。"""

    def __init__(self, records=()):
        self._records = list(records)
        self.purged: list[str] = []
        self.history_available = True

    def discover(self):
        return list(self._records)

    def purge(self, thread_id):
        self.purged.append(thread_id)

    def restore_payload(self, record):
        return RestorePayload(
            status=record.status, events=record.events, summary=record.summary
        )


def test_reaped_session_stays_in_the_listing(tmp_path):
    """**回收 ≠ 历史消失**：空闲回收把会话摘出内存后，列表里必须还有它。

    这正是用户最初那个问题换了个触发点：P7 的注册表只有一本内存账，回收即 404；
    目录改从持久化读之后，「回收」必须留一条未物化记录（从内存现成的状态造，
    **不查库** —— sqlite 下事件从不落库，不带上就永远读不回来了）。
    """
    catalog = _FakeCatalog()
    clock = [1000.0]
    registry = SessionRegistry(
        _sqlite_settings(tmp_path),
        make_llm=_llm_factory(),
        require_approval_for=(),
        catalog=catalog,
        clock=lambda: clock[0],
        idle_timeout=30.0,
    )
    session = registry.create("reapme")
    assert session.begin("写快排") is True
    assert _wait_until(lambda: session.status is SessionStatus.IDLE)
    event_count = len(session.events)
    assert event_count > 0

    clock[0] += 100.0
    assert registry.reap_idle() == ["reapme"]

    assert "reapme" not in registry, "内存里必须已经放掉了（容器/图都关了）"
    snapshot = next(s for s in registry.snapshots() if s["thread_id"] == "reapme")
    assert snapshot["status"] == "idle"
    assert snapshot["event_count"] == event_count, "回收的记录必须带着内存里的事件"
    assert snapshot["result"] is not None, "结果摘要也要留下（它是用户唯一能看到的收尾）"


def test_reaped_session_can_be_deleted_for_real(tmp_path):
    """回收之后仍然删得掉，而且是**连库一起删**（否则「回收过的会话」是删不掉的僵尸）。"""
    catalog = _FakeCatalog()
    clock = [1000.0]
    registry = SessionRegistry(
        _sqlite_settings(tmp_path),
        make_llm=_llm_factory(),
        require_approval_for=(),
        catalog=catalog,
        clock=lambda: clock[0],
        idle_timeout=30.0,
    )
    session = registry.create("reapme")
    assert session.begin("写快排") is True
    assert _wait_until(lambda: session.status is SessionStatus.IDLE)
    clock[0] += 100.0
    registry.reap_idle()

    assert registry.close("reapme") is True
    assert catalog.purged == ["reapme"]
    assert registry.snapshots() == []


def test_registry_without_catalog_does_not_remember_reaped_sessions(tmp_path):
    """没有 catalog（纯内存模式）= P7 行为：回收即消失，`len`/`in` 也只数内存。

    这条是**容器协议的护栏**：`__len__`/`__contains__` 的语义是「注册表还握着没有」，
    `test_session.py` 的 `"idle" not in reg`、`len(reg) == 0` 全靠它。把未物化记录
    也算进去会静默破坏那些断言（我第一版就是这么错的）。
    """
    clock = [1000.0]
    registry = SessionRegistry(
        _sqlite_settings(tmp_path),
        make_llm=_llm_factory(),
        require_approval_for=(),
        clock=lambda: clock[0],
        idle_timeout=30.0,
    )
    session = registry.create("plain")
    assert session.begin("写快排") is True
    assert _wait_until(lambda: session.status is SessionStatus.IDLE)

    clock[0] += 100.0
    registry.reap_idle()

    assert len(registry) == 0
    assert "plain" not in registry
    assert registry.snapshots() == []
    assert registry.history_available is False


# ---------------------------------------------------------------- 真 PG：事件流跨重启

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")


def _pg_ready() -> bool:
    """库可达才跑（连不上自动 skip，与 `test_postgres_live.py` 同一判据同一 env）。

    **刻意读 TEST_DATABASE_URL 而不是 DATABASE_URL**：这些用例会真往库里写会话与事件，
    不该因为开发者本机 `.env` 设了 DATABASE_URL 就顺手污染他正在用的库。
    """
    if not TEST_DATABASE_URL:
        return False
    try:
        import psycopg
    except ImportError:  # pragma: no cover
        return False
    try:
        with psycopg.connect(TEST_DATABASE_URL, connect_timeout=3):
            return True
    except Exception:  # noqa: BLE001
        return False


pg_only = pytest.mark.skipif(
    not _pg_ready(),
    reason="TEST_DATABASE_URL 未设置或 PostgreSQL 不可达（先 docker compose up -d）",
)


def _pg_settings(tmp_path) -> Settings:
    return Settings(
        llm_api_key="fake-key",
        llm_base_url="",
        llm_model="fake",
        workspace_root=Path(tmp_path) / "ws",
        checkpoint_db_path=Path(tmp_path) / "unused.db",  # postgres 后端下不使用
        persistence_backend="postgres",
        database_url=TEST_DATABASE_URL,
        session_idle_timeout=1800.0,
    )


def _pg_thread_id() -> str:
    """每个用例一个独立 thread_id：互不干扰、无需清表（events/checkpoints 都按 thread 分区）。"""
    return "p9-test-" + uuid.uuid4().hex[:12]


@pg_only
def test_pg_restart_replays_the_full_event_stream(tmp_path):
    """sqlite 只列空壳；**PG 必须把整段历史还回来**（`seq` 从 0 连续、逐条对得上）。

    这是 P9 存在的理由：用户重启后发现「库里明明有 138 条事件，前端一条都看不到」。
    断言按 seq 而不是按时间戳 —— seq 是前端折叠算法的坐标系（`?since` 游标、
    `if (env.seq <= lastSeq) return` 的判重），它错了整个时间线都是错的。
    """
    settings = _pg_settings(tmp_path)
    tid = _pg_thread_id()
    try:
        with _client(settings, scripts=DELEGATE_TO_CODER) as client:
            _create(client, tid)
            _start(client, tid)
            _await_events_settled(client, tid)
            live = _events(client, tid)
        assert len(live) >= 5, f"事件太少，这条用例会失去意义: {len(live)}"
        assert [e["seq"] for e in live] == list(range(len(live)))

        # ------------------------------ 重启
        with _client(settings) as client:
            body = _list(client)
            assert body["history_available"] is True

            info = next(s for s in body["sessions"] if s["thread_id"] == tid)
            assert info["status"] == "idle"  # finished → 可续跑（③）
            assert info["event_count"] == len(live)

            after = _events(client, tid)
            assert [e["seq"] for e in after] == list(range(len(live)))
            assert [e["event"]["type"] for e in after] == [e["event"]["type"] for e in live]
            assert [e["event"]["message"] for e in after] == [
                e["event"]["message"] for e in live
            ]

            # `?since` 是闭区间 → 断线重连传的是 lastSeq+1；这里验最后一条。
            tail = _events(client, tid, since=len(live) - 1)
            assert [e["seq"] for e in tail] == [len(live) - 1]

        # ------------------------------ 再重启一次并**续跑**：seq 必须接着往下走
        with _client(settings, scripts=SECOND_ROUND) as client:
            _start(client, tid, "再写一个 util.py")
            _await_new_events_settled(client, tid, before=len(live))
            continued = _events(client, tid)
            assert [e["seq"] for e in continued] == list(range(len(continued)))
            assert len(continued) > len(live), "续跑必须产生新事件"
            assert _workspace(tmp_path, tid, "util.py").read_text(encoding="utf-8") == "X = 1\n"

        # ------------------------------ ⑤ DELETE 连库一起删：重启不复现
        with _client(settings) as client:
            assert client.delete(f"/sessions/{tid}").status_code == 204
        with _client(settings) as client:
            assert tid not in _ids(client)
            assert client.get(f"/sessions/{tid}").status_code == 404
    finally:
        _pg_cleanup(tid)


@pg_only
def test_pg_restart_keeps_awaiting_approval_across_processes(tmp_path):
    """⑥ 的 PG 版：**跨进程 resume 在真数据库上真的能把活干完**。

    真机 spike 已实证（全新 Python 进程 → `Command(resume="yes")` → 文件落盘），
    这条把它固化成回归：走 HTTP 端点、走 PG 的 `checkpoint_writes`。
    """
    settings = _pg_settings(tmp_path)
    tid = _pg_thread_id()
    try:
        with _client(settings, scripts=DELEGATE_TO_CODER, approvals=("coder",)) as client:
            _create(client, tid)
            _start(client, tid)
            _await_status(client, tid, "awaiting_approval")
            assert not _workspace(tmp_path, tid, "main.py").exists()

        with _client(settings, scripts=RESUME_TAIL, approvals=("coder",)) as client:
            info = _info(client, tid)
            assert info["status"] == "awaiting_approval"
            assert info["approval"][0]["specialist"] == "coder"

            response = client.post(f"/sessions/{tid}/approval", json={"approved": True})
            assert response.status_code == 200, response.text
            _await_status(client, tid, "idle")

            assert _workspace(tmp_path, tid, "main.py").read_text(encoding="utf-8") == QUICKSORT
    finally:
        _pg_cleanup(tid)


def _pg_cleanup(thread_id: str) -> None:
    """把测试会话从库里彻底擦掉（**用例失败在半路时也要擦**，不给用户的库留垃圾）。

    直接用 catalog 的 `purge`（正是被测的那段代码）而不是走 HTTP：用例可能死在
    「app 都没起来」的阶段，那时没有服务可打。擦不掉只告警 —— 绝不能让它盖掉
    用例真正的失败原因。
    """
    from persistence.event_store import PostgresEventStore
    from persistence.pool import build_pool, close_pool
    from runtime.catalog import CatalogError, SessionCatalog

    placeholder = Path("unused-for-cleanup")
    settings = Settings(
        llm_api_key="",
        llm_base_url="",
        llm_model="fake",
        workspace_root=placeholder,
        checkpoint_db_path=placeholder / "unused.db",
        persistence_backend="postgres",
        database_url=TEST_DATABASE_URL,
    )
    try:
        pool = build_pool(settings)
        store = PostgresEventStore(settings.database_url, pool=pool)
        store.open()
        try:
            with SessionCatalog(settings, pool=pool, event_store=store) as catalog:
                catalog.purge(thread_id)
        finally:
            store.close()
            close_pool(pool)
    except CatalogError as e:  # 删不掉：真实失败原因优先，这里只留个声
        print(f"警告: 清理 PG 测试会话 {thread_id} 失败: {e}")
