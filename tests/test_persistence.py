"""P1: SqliteSaver 持久化 —— 落盘、关库重开恢复、多会话隔离、续跑。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_agent_graph
from tools.registry import build_tools
from workspace.manager import WorkspaceManager


class FakeLLM:
    """按预设脚本吐 AIMessage，确定性驱动 ReAct 循环（同 test_agent_loop）。"""

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


def _initial(task="写文件"):
    return {
        "messages": [SystemMessage(content="s"), HumanMessage(content=task)],
        "current_agent": "Coder",
        "current_task": task,
        "iteration_count": 0,
        "max_iterations": 10,
        "status": "running",
        "tool_calls": [],
        "observations": [],
    }


def _run_task(db_path, ws_root, script, thread_id="t1"):
    """在独立 sqlite 库上跑一次完整任务，返回 (result, fake, checkpoint数)。"""
    ws = WorkspaceManager(ws_root)
    fake = FakeLLM(script)
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_agent_graph(fake, build_tools(ws), checkpointer=saver)
        result = graph.invoke(_initial(), {"configurable": {"thread_id": thread_id}})
        n_ckpts = len(list(graph.get_state_history({"configurable": {"thread_id": thread_id}})))
    return result, fake, n_ckpts


def test_sqlite_roundtrip_recovers_full_state(tmp_path):
    db = tmp_path / "checkpoints.db"
    ws_root = tmp_path / "ws"

    result, fake, n_ckpts = _run_task(
        db,
        ws_root,
        [
            _tool_call(1, "write_file", {"path": "a.py", "content": "x = 1\n"}),
            AIMessage(content="写好了"),
        ],
    )
    assert result["status"] == "finished"
    assert fake.calls == 2
    assert n_ckpts >= 1  # 每个 superstep 都落盘

    # 关库后重开：状态必须完整恢复
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_agent_graph(
            FakeLLM([]), build_tools(WorkspaceManager(ws_root)), checkpointer=saver
        )
        st = graph.get_state({"configurable": {"thread_id": "t1"}})
        msgs = st.values["messages"]
        # sys + human + tool_call + ToolMessage + final = 5
        assert len(msgs) == 5
        assert "写好了" in msgs[-1].content


def test_resume_can_continue_on_recovered_history(tmp_path):
    db = tmp_path / "checkpoints.db"
    ws_root = tmp_path / "ws"
    _run_task(
        db,
        ws_root,
        [
            _tool_call(1, "write_file", {"path": "a.py", "content": "x = 1\n"}),
            AIMessage(content="写好了"),
        ],
    )

    # 重开库，在已有历史上追加一条新指令，应继续执行
    fake = FakeLLM([AIMessage(content="继续完成")])
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_agent_graph(
            fake, build_tools(WorkspaceManager(ws_root)), checkpointer=saver
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="再写一个 b.py")]},
            {"configurable": {"thread_id": "t1"}},
        )
        assert result["status"] == "finished"
        assert fake.calls == 1  # 恢复后基于历史继续决策，而不是重跑
        st = graph.get_state({"configurable": {"thread_id": "t1"}})
        assert len(st.values["messages"]) == 7  # 5 + 新 human + 新 agent


def test_thread_isolation(tmp_path):
    db = tmp_path / "checkpoints.db"
    ws_root = tmp_path / "ws"
    _run_task(
        db,
        ws_root,
        [
            _tool_call(1, "write_file", {"path": "a.py", "content": "x = 1\n"}),
            AIMessage(content="写好了"),
        ],
        thread_id="t1",
    )

    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_agent_graph(
            FakeLLM([]), build_tools(WorkspaceManager(ws_root)), checkpointer=saver
        )
        # 新会话是空状态
        st_new = graph.get_state({"configurable": {"thread_id": "t2"}})
        assert len(st_new.values.get("messages", [])) == 0
        # 原会话不受影响
        st_old = graph.get_state({"configurable": {"thread_id": "t1"}})
        assert len(st_old.values["messages"]) == 5
