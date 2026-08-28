"""P4-3: Condense 测试 —— 长会话 messages 压缩。

覆盖：触发压缩（头/尾保留）、低于阈值不触发（回归）、最近 tool_call↔ToolMessage
配对完整、supervisor 委派链兼容、interrupt/resume 兼容。
P4-3-2 追加：token 守卫（tiktoken 注入点）——消息少但 token 巨大触发、context_limit=None
关闭、reserve_tokens 边界、消息数触发保留、事件带 token 计数。
"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from agent.condense import make_condense_node
from agent.graph import build_agent_graph
from agent.specialists import build_supervisor_prompt
from agent.supervisor import build_supervisor_graph
from conftest import FakeLLM, QUICKSORT_CODE, _tool_call
from config.settings import Settings
from events.events import EventType, emitter, format_event
from tools.registry import build_tools
from workspace.manager import WorkspaceManager


def _settings(tmp_path) -> Settings:
    return Settings(
        llm_api_key="",
        llm_base_url="",
        llm_model="fake",
        workspace_root=tmp_path / "ws",
        max_iterations=10,
    )


def _initial(task="condense 测试任务"):
    return {
        "messages": [
            SystemMessage(content=build_supervisor_prompt()),
            HumanMessage(content=task),
        ],
        "current_agent": "Supervisor",
        "current_task": task,
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
        "tool_calls": [],
        "observations": [],
    }


def _make_fakes(scripts: dict[str, list]) -> tuple:
    fakes = {role: FakeLLM(script) for role, script in scripts.items()}

    def make(role: str) -> FakeLLM:
        return fakes.setdefault(role, FakeLLM([]))

    return make, fakes


def _summaries(messages) -> list:
    return [m for m in messages if m.type == "system" and "历史压缩" in m.content]


def _trigger_events() -> list:
    return [e for e in emitter.events if e.type is EventType.CONDENSE]


def _pair_state(n_pairs: int, big_content: int = 0, big_index: int = 0) -> list:
    """构造 n_pairs 个完整 AI(带 tool_call)↔ToolMessage 对（显式 id，RemoveMessage 可用）。

    big_content>0 时把第 big_index 对（默认第 0 对，落进被压缩的中间区）的工具结果放大，
    制造「消息少但 token 巨大」的 token 守卫场景。
    """
    msgs = [
        SystemMessage(content="sys", id="h0"),
        HumanMessage(content="task", id="h1"),
    ]
    for i in range(n_pairs):
        content = "x" * big_content if (big_content and i == big_index) else "ok"
        msgs.append(
            AIMessage(
                content="",
                id=f"a{i}",
                tool_calls=[{"name": "list_files", "args": {}, "id": f"call_{i}", "type": "tool_call"}],
            )
        )
        msgs.append(
            ToolMessage(content=content, tool_call_id=f"call_{i}", name="list_files", id=f"r{i}")
        )
    return msgs


def test_condense_triggers_and_preserves_head_recent(tmp_path):
    """P4-3: 超阈值触发压缩——头部(sys+task)保留、历史被删、含摘要、循环正常收尾。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    tools = build_tools(ws)
    script = [
        _tool_call(1, "list_files", {}),
        _tool_call(2, "list_files", {}),
        _tool_call(3, "list_files", {}),
        AIMessage(content="完成"),
    ]
    fake = FakeLLM(script)
    graph = build_agent_graph(
        fake, tools, condense_node=make_condense_node(trigger_count=3, keep_recent=2)
    )
    initial = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="task")],
        "current_agent": "Coder",
        "current_task": "task",
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
    }
    result = graph.invoke(initial, {"configurable": {"thread_id": "c1"}})

    assert result["status"] == "finished"
    msgs = result["messages"]
    # 头部（系统提示 + 任务）保留
    assert msgs[0].type == "system" and msgs[1].type == "human"
    # 触发过压缩：至少一条 "历史压缩" 摘要
    assert _summaries(msgs)
    # 无压缩时本应是 2 + 3*2 + 1 = 9 条；压缩后严格更少
    assert len(msgs) < 9


def test_condense_noop_below_threshold(tmp_path):
    """P4-3: 低于阈值不触发——messages 无删减、无摘要，行为与不加该节点一致。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    tools = build_tools(ws)
    script = [
        _tool_call(1, "list_files", {}),
        AIMessage(content="完成"),
    ]
    fake = FakeLLM(script)
    graph = build_agent_graph(
        fake, tools, condense_node=make_condense_node(trigger_count=100, keep_recent=2)
    )
    initial = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="task")],
        "current_agent": "Coder",
        "current_task": "task",
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
    }
    result = graph.invoke(initial, {"configurable": {"thread_id": "c2"}})

    assert result["status"] == "finished"
    msgs = result["messages"]
    # sys, task, AIMessage(list_files), ToolMessage, 完成 = 5 条，一条不少
    assert len(msgs) == 5
    assert not _summaries(msgs)


def test_condense_keeps_latest_tool_pair(tmp_path):
    """P4-3: 压缩后最近一次 tool_call↔ToolMessage 配对仍在 recent 窗口且相邻完整。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    tools = build_tools(ws)
    script = [
        _tool_call(1, "list_files", {}),
        _tool_call(2, "list_files", {}),
        _tool_call(3, "list_files", {}),
        AIMessage(content="完成"),
    ]
    fake = FakeLLM(script)
    graph = build_agent_graph(
        fake, tools, condense_node=make_condense_node(trigger_count=3, keep_recent=2)
    )
    initial = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="task")],
        "current_agent": "Coder",
        "current_task": "task",
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
    }
    result = graph.invoke(initial, {"configurable": {"thread_id": "c3"}})

    msgs = result["messages"]
    tool_ais = [m for m in msgs if getattr(m, "tool_calls", None)]
    assert tool_ais, "最近一次工具决策的 AIMessage 不能被压缩掉"
    last = tool_ais[-1]
    call_id = last.tool_calls[0]["id"]
    # 其 ToolMessage 紧跟其后（recent 窗口内相邻），配对完整
    idx = msgs.index(last)
    assert msgs[idx + 1].type == "tool"
    assert msgs[idx + 1].tool_call_id == call_id


def test_condense_supervisor_delegate_chain(tmp_path):
    """P4-3: supervisor 多次委派超阈值触发压缩——delegate 机制不受影响、文件落盘。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="快排已写入 main.py"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "coder", "task": "写快排"}),
        _tool_call(2, "delegate", {"specialist": "coder", "task": "写快排 v2"}),
        _tool_call(3, "delegate", {"specialist": "coder", "task": "写快排 v3"}),
        AIMessage(content="完成"),
    ])

    graph = build_supervisor_graph(
        _settings(tmp_path), ws, make_llm=make, require_approval_for=(),
        condense_node=make_condense_node(trigger_count=3, keep_recent=2),
    )
    result = graph.invoke(_initial(), {"configurable": {"thread_id": "c4"}})

    assert result["status"] == "finished"
    assert "quicksort" in ws.read_text("main.py")  # 子 agent 真实写盘
    # 压缩触发过：supervisor 历史里有 "历史压缩" 摘要
    assert _summaries(result["messages"])
    # 委派结果 ToolMessage 仍有（压缩只把中间旧历史收进摘要，recent 内保留最新）
    tmsgs = [m for m in result["messages"] if m.type == "tool"]
    assert tmsgs


def test_condense_emits_event_on_trigger_not_on_noop(tmp_path):
    """P4-3: 触发压缩时发 CONDENSE 事件（带 before/after/removed），低于阈值不发。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    tools = build_tools(ws)
    initial = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="task")],
        "current_agent": "Coder",
        "current_task": "task",
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
    }

    # 低于阈值 → 不触发 → 无 CONDENSE 事件
    noop = build_agent_graph(
        FakeLLM([_tool_call(1, "list_files", {}), AIMessage(content="完成")]),
        tools,
        condense_node=make_condense_node(trigger_count=100, keep_recent=2),
    )
    noop.invoke(initial, {"configurable": {"thread_id": "e1"}})
    assert not [e for e in emitter.events if e.type is EventType.CONDENSE]

    # 触发 → CONDENSE 事件携带触发时机与结果（before/after/removed 关系自洽）
    emitter.clear()
    trig = build_agent_graph(
        FakeLLM([
            _tool_call(1, "list_files", {}),
            _tool_call(2, "list_files", {}),
            _tool_call(3, "list_files", {}),
            AIMessage(content="完成"),
        ]),
        tools,
        condense_node=make_condense_node(trigger_count=3, keep_recent=2),
    )
    trig.invoke(initial, {"configurable": {"thread_id": "e2"}})
    cond = [e for e in emitter.events if e.type is EventType.CONDENSE]
    assert cond, "触发路径应有 CONDENSE 事件"
    for e in cond:
        assert e.agent == "Coder"
        assert e.detail["removed"] >= 1
        assert e.detail["after"] == e.detail["before"] - e.detail["removed"] + 1
        # 具体压缩结果：摘要内容随事件携带，且 CLI 渲染能看到
        assert e.detail["summary"].startswith("历史压缩")
        rendered = format_event(e)
        assert "[Condense]" in rendered and "压缩历史" in rendered and "历史压缩" in rendered


def test_condense_interrupt_resume_compatible(tmp_path):
    """P4-3: condense 与 Human Approval interrupt/resume 兼容——挂起→恢复→完成。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    make, fakes = _make_fakes({
        "analyst": [AIMessage(content="分析完成")],
        "coder": [
            _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
            AIMessage(content="快排已写入"),
        ],
    })
    fakes["Supervisor"] = FakeLLM([
        _tool_call(1, "delegate", {"specialist": "analyst", "task": "分析现状"}),
        _tool_call(2, "delegate", {"specialist": "analyst", "task": "再分析一次"}),
        _tool_call(3, "delegate", {"specialist": "coder", "task": "写快排"}),
        AIMessage(content="完成"),
    ])

    # 默认 require_approval_for=("coder",)：前 2 次 analyst（只读）不 interrupt，
    # 第 3 次 coder 委派时 interrupt；condense 已在前面轮次触发过。
    graph = build_supervisor_graph(
        _settings(tmp_path), ws, make_llm=make,
        condense_node=make_condense_node(trigger_count=3, keep_recent=2),
    )
    config = {"configurable": {"thread_id": "c5"}}
    graph.invoke(_initial(), config)

    # 挂在 coder 的 interrupt 上（condense 节点未影响挂起时序）
    snap = graph.get_state(config)
    assert snap.next == ("tools",)
    assert len(snap.interrupts) == 1
    assert snap.interrupts[0].value["specialist"] == "coder"

    result = graph.invoke(Command(resume="yes"), config)
    assert result["status"] == "finished"
    assert "quicksort" in ws.read_text("main.py")
    # 恢复路径上 condense 正常（历史含摘要）
    assert _summaries(result["messages"])


# ---- P4-3-2: Token-aware Context Budget ----

def test_token_trigger_preserves_latest_pair(tmp_path):
    """P4-3-2: token 守卫触发路径——读大文件产生巨型工具结果，消息数远低于阈值仍压缩，
    且最新 AI↔ToolMessage 配对保持相邻完整。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    ws.write_text("big.txt", "z" * 3000)
    tools = build_tools(ws)
    script = [
        _tool_call(1, "read_file", {"path": "big.txt"}),
        _tool_call(2, "read_file", {"path": "big.txt"}),
        _tool_call(3, "read_file", {"path": "big.txt"}),
        AIMessage(content="完成"),
    ]
    fake = FakeLLM(script)
    graph = build_agent_graph(
        fake, tools,
        condense_node=make_condense_node(
            trigger_count=100,  # 消息数守卫关掉（最多 ~8 条）
            keep_recent=2,
            context_limit=2000,  # token 预算小 → 每轮巨型工具结果都超
            reserve_tokens=0,
            count_tokens_fn=lambda s: len(s),
        ),
    )
    initial = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="task")],
        "current_agent": "Coder",
        "current_task": "task",
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
    }
    result = graph.invoke(initial, {"configurable": {"thread_id": "t1"}})

    assert _trigger_events(), "巨型工具结果应触发 token 守卫"
    msgs = result["messages"]
    assert _summaries(msgs)
    # 最新 AI↔ToolMessage 配对在 recent 窗口内相邻完整（token 触发路径同样保证）
    tool_ais = [m for m in msgs if getattr(m, "tool_calls", None)]
    assert tool_ais
    last = tool_ais[-1]
    call_id = last.tool_calls[0]["id"]
    idx = msgs.index(last)
    assert msgs[idx + 1].type == "tool"
    assert msgs[idx + 1].tool_call_id == call_id


def test_context_limit_none_disables_token_guard():
    """P4-3-2: context_limit=None → token 守卫关闭——消息少但 token 巨大不触发（回归锚点）。"""
    emitter.clear()
    msgs = _pair_state(6, big_content=5000)  # 14 条，单条工具结果 5000 字符
    node = make_condense_node(
        trigger_count=100, keep_recent=2, count_tokens_fn=lambda s: len(s)
    )
    updates = node({"messages": msgs, "current_agent": "Coder"})
    assert updates == {}
    assert not _trigger_events()


def test_reserve_tokens_shifts_threshold():
    """P4-3-2: reserve_tokens 把触发阈值从 context_limit 往下挪——同一组消息同一预算，
    reserve 足够大时从不触发变触发（边界验证）。"""
    counter = lambda s: len(s)  # noqa: E731
    msgs = _pair_state(6)  # 估算恒 = 19（sys=3 + task=4 + 6*Tool("ok")=12）
    no_reserve = make_condense_node(
        trigger_count=10**9, keep_recent=2,
        context_limit=29, reserve_tokens=0, count_tokens_fn=counter,
    )
    with_reserve = make_condense_node(
        trigger_count=10**9, keep_recent=2,
        context_limit=29, reserve_tokens=11, count_tokens_fn=counter,
    )
    # reserve=0 → 阈值 29 ≥ 19，不触发
    assert no_reserve({"messages": msgs, "current_agent": "Coder"}) == {}
    # reserve=11 → 阈值 18 < 19，触发
    emitter.clear()
    assert with_reserve({"messages": msgs, "current_agent": "Coder"}) != {}
    assert _trigger_events()


def test_message_count_trigger_preserved_with_token_guard():
    """P4-3-2: token 守卫开启也不改变消息数触发——消息多但 token 少仍由消息数触发。"""
    emitter.clear()
    msgs = _pair_state(6)  # 14 条（> trigger_count=10），token 很少
    node = make_condense_node(
        trigger_count=10, keep_recent=2,
        context_limit=10**9, reserve_tokens=0, count_tokens_fn=lambda s: len(s),
    )
    updates = node({"messages": msgs, "current_agent": "Coder"})
    assert updates != {}
    e = _trigger_events()[-1]
    assert "触发: 消息数" in e.message
    assert "触发: token" not in e.message


def test_token_condense_event_reports_token_counts():
    """P4-3-2: token 触发的事件 detail 带 before_tokens/after_tokens，压缩后明显下降且 CLI 可见。"""
    emitter.clear()
    msgs = _pair_state(6, big_content=5000, big_index=0)  # 大内容在被压缩区 → 压缩后 token 大降
    node = make_condense_node(
        trigger_count=10**9, keep_recent=2,
        context_limit=1000, reserve_tokens=0, count_tokens_fn=lambda s: len(s),
    )
    node({"messages": msgs, "current_agent": "Coder"})
    cond = _trigger_events()
    assert cond
    e = cond[-1]
    assert e.detail["before_tokens"] > 1000
    assert e.detail["after_tokens"] < e.detail["before_tokens"]
    rendered = format_event(e)
    assert "压缩历史" in rendered and "token" in rendered
