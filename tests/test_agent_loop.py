"""用脚本化 FakeLLM 确定性跑通完整 ReAct 闭环（不需要 API key / 网络）。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph import build_agent_graph
from conftest import FakeLLM, QUICKSORT_CODE, TEST_CODE, _tool_call
from events.events import EventType, emitter, format_event
from tools.registry import build_tools
from workspace.manager import WorkspaceManager


def test_react_loop_writes_files_and_runs_test(tmp_path):
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    tools = build_tools(ws)

    script = [
        _tool_call(1, "write_file", {"path": "main.py", "content": QUICKSORT_CODE}),
        _tool_call(2, "write_file", {"path": "test_main.py", "content": TEST_CODE}),
        _tool_call(3, "run_command", {"command": "python -m pytest -q"}),
        AIMessage(content="已完成：创建了 quicksort 及测试，pytest 全部通过。"),
    ]
    fake = FakeLLM(script)
    graph = build_agent_graph(fake, tools)

    initial = {
        "messages": [
            SystemMessage(content="sys"),
            HumanMessage(content="创建快排和测试"),
        ],
        "current_agent": "Coder",
        "current_task": "创建快排和测试",
        "iteration_count": 0,
        "max_iterations": 20,
        "status": "running",
        "tool_calls": [],
        "observations": [],
    }

    result = graph.invoke(initial, {"configurable": {"thread_id": "t1"}})

    # 1) 循环正常结束
    assert result["status"] == "finished"
    assert "已完成" in result["result"]

    # 2) 文件真的被写进了 workspace
    assert "quicksort" in ws.read_text("main.py")
    assert "test_quicksort" in ws.read_text("test_main.py")

    # 3) LLM 被调用了 4 次：3 次工具决策 + 1 次最终回答
    assert fake.calls == 4

    # 4) 事件轨迹记录了 3 次工具调用完成（Agent→Tool→Result 往返可见）
    completed = [e for e in emitter.events if e.type is EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 3


def test_unknown_tool_returns_error_and_continues(tmp_path):
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    tools = build_tools(ws)

    script = [
        _tool_call(1, "no_such_tool", {}),
        AIMessage(content="工具不存在，我改用别的方式完成任务。"),
    ]
    fake = FakeLLM(script)
    graph = build_agent_graph(fake, tools)

    initial = {
        "messages": [SystemMessage(content="s"), HumanMessage(content="t")],
        "current_agent": "Coder",
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
    }
    result = graph.invoke(initial, {"configurable": {"thread_id": "t2"}})

    # 未知工具 -> 回写错误 ToolMessage -> agent 继续决策 -> 正常结束
    assert result["status"] == "finished"
    failed = [e for e in emitter.events if e.type is EventType.TOOL_CALL_FAILED]
    assert len(failed) == 1


def test_real_token_usage_emits_per_call(tmp_path):
    """P4-3-2 可观测：真实 LLM 响应带 token_usage 时，每次调用发 TOKEN_USAGE 事件，CLI 可见。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    tools = build_tools(ws)
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    script = [
        AIMessage(
            content="",
            tool_calls=[{"name": "list_files", "args": {}, "id": "call_1", "type": "tool_call"}],
            response_metadata={"token_usage": usage},
        ),
        AIMessage(content="完成", response_metadata={"token_usage": usage}),
    ]
    fake = FakeLLM(script)
    graph = build_agent_graph(fake, tools)
    initial = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="task")],
        "current_agent": "Coder",
        "current_task": "task",
        "iteration_count": 0,
        "max_iterations": 20,
        "status": "running",
        "tool_calls": [],
        "observations": [],
    }
    result = graph.invoke(initial, {"configurable": {"thread_id": "u1"}})

    assert result["status"] == "finished"
    usage_events = [e for e in emitter.events if e.type is EventType.TOKEN_USAGE]
    # 每次 LLM 调用发一次：1 次工具决策 + 1 次最终回答
    assert len(usage_events) == fake.calls == 2
    for e in usage_events:
        assert e.agent == "Coder"
        assert e.detail == usage
        rendered = format_event(e)
        assert "消耗 120 tokens" in rendered and "输入 100" in rendered


def test_no_token_usage_without_response_metadata(tmp_path):
    """P4-3-2 可观测：FakeLLM 无 response_metadata → 不发 TOKEN_USAGE（只有真实 LLM 有 usage）。"""
    emitter.clear()
    ws = WorkspaceManager(tmp_path / "ws")
    tools = build_tools(ws)
    graph = build_agent_graph(
        FakeLLM([_tool_call(1, "list_files", {}), AIMessage(content="完成")]), tools
    )
    initial = {
        "messages": [SystemMessage(content="sys"), HumanMessage(content="task")],
        "current_agent": "Coder",
        "current_task": "task",
        "iteration_count": 0,
        "max_iterations": 20,
        "status": "running",
    }
    graph.invoke(initial, {"configurable": {"thread_id": "u2"}})
    assert not [e for e in emitter.events if e.type is EventType.TOKEN_USAGE]
