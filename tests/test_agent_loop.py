"""用脚本化 FakeLLM 确定性跑通完整 ReAct 闭环（不需要 API key / 网络）。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.graph import build_agent_graph
from events.events import EventType, emitter
from tools.registry import build_tools
from workspace.manager import WorkspaceManager


class FakeLLM:
    """按预设脚本吐 AIMessage，确定性驱动 ReAct 循环。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if not self.script:
            return AIMessage(content="(fallback done)")
        return self.script.pop(0)


def _tool_call(idx, name, args):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{idx}", "type": "tool_call"}],
    )


QUICKSORT_CODE = (
    "def quicksort(arr):\n"
    "    if len(arr) <= 1:\n"
    "        return arr\n"
    "    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n"
    "    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)\n"
)
TEST_CODE = (
    "from main import quicksort\n\n"
    "def test_quicksort():\n"
    "    assert quicksort([3, 1, 2]) == [1, 2, 3]\n"
)


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
