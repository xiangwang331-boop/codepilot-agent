"""CodePilot CLI 入口。

用法：
    python main.py "创建一个 Python 快速排序程序，并创建单元测试"   # 新任务，自动生成会话 ID
    python main.py --thread my-task "新任务，指定会话 ID"
    python main.py --resume <会话ID> "可选的新指令"                 # 恢复上次中断的任务
    python main.py --resume <会话ID>                                # 直接续跑（不带新指令）
或直接运行后粘贴需求。

需要环境变量（见 .env.example）：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL。
checkpoint 落在 CHECKPOINT_DB_PATH（默认 data/checkpoints.db，SQLite）。
"""
from __future__ import annotations

import argparse
import sys
import uuid

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出，避免中文乱码
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.specialists import build_supervisor_prompt
from agent.supervisor import build_supervisor_graph
from config.settings import Settings
from events.events import EventType, emit, emitter, format_event
from workspace.manager import WorkspaceManager


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="codepilot",
        description="LangGraph Multi-Agent 软件工程运行时 CLI",
    )
    parser.add_argument(
        "task", nargs="*", help="开发需求（可多词，如 \"创建快排和测试\"）"
    )
    parser.add_argument(
        "--thread",
        metavar="ID",
        help="指定会话 ID（默认自动生成 UUID）",
    )
    parser.add_argument(
        "--resume",
        metavar="ID",
        help="恢复指定会话：从上次 checkpoint 续跑，可再追加一条新指令",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    task = " ".join(args.task).strip()
    resume_mode = bool(args.resume)
    thread_id = args.resume or args.thread or uuid.uuid4().hex

    if not resume_mode and not task:
        print("错误：请提供开发需求，或用 --resume <会话ID> 恢复任务。")
        print("  示例: python main.py \"创建快排和测试\"")
        print("        python main.py --resume <会话ID>")
        return

    settings = Settings.from_env()

    if not settings.llm_api_key:
        print("错误：未设置 LLM_API_KEY（环境变量）。")
        print("  示例(PowerShell):")
        print("    $env:LLM_API_KEY = 'sk-...'")
        print("    $env:LLM_BASE_URL = 'https://api.deepseek.com/v1'")
        print("    $env:LLM_MODEL = 'deepseek-chat'")
        return

    # SQLite 不会自动创建父目录
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    ws = WorkspaceManager(settings.workspace_root)

    # 实时打印事件
    emitter.add_listener(lambda e: print(format_event(e)))

    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        graph = build_supervisor_graph(settings, ws, checkpointer=checkpointer)

        if resume_mode:
            # 恢复：先确认会话存在，再决定是否追加新指令
            prev = graph.get_state(config)
            n_msgs = len(prev.values.get("messages", []))
            if n_msgs == 0:
                print(f"错误：会话 {thread_id} 不存在或已清理。")
                return
            print(f"\n=== 恢复会话 {thread_id}（已有 {n_msgs} 条消息）===")
            if task:
                print(f"追加指令: {task}")
                graph_input = {"messages": [HumanMessage(content=task)]}
            else:
                graph_input = None
        else:
            emit(EventType.AGENT_STARTED, agent="Supervisor", message="")
            print(f"\n=== 任务 ===\n{task}\n")
            print(f"会话 ID: {thread_id}")
            graph_input = {
                "messages": [
                    SystemMessage(content=build_supervisor_prompt()),
                    HumanMessage(content=task),
                ],
                "current_agent": "Supervisor",
                "current_task": task,
                "iteration_count": 0,
                "max_iterations": settings.max_iterations,
                "status": "running",
                "tool_calls": [],
                "observations": [],
            }

        result = graph.invoke(graph_input, config)

    print("\n=== 结果 ===")
    if result.get("status") == "finished":
        emit(EventType.AGENT_COMPLETED, agent="Supervisor", message="")
        print(result.get("result") or "(无内容)")
    else:
        emit(
            EventType.AGENT_FAILED,
            agent="Supervisor",
            message=result.get("error") or f"状态 {result.get('status')}",
        )
        print(f"状态: {result.get('status')} — {result.get('error') or ''}")


if __name__ == "__main__":
    main()
