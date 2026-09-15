"""CodePilot CLI 入口。

用法：
    python main.py "创建一个 Python 快速排序程序，并创建单元测试"   # 新任务，自动生成会话 ID
    python main.py --thread my-task "新任务，指定会话 ID"
    python main.py --resume <会话ID> "可选的新指令"                 # 恢复上次中断的任务
    python main.py --resume <会话ID>                                # 直接续跑（不带新指令）
    python main.py --events <会话ID>                                # 回放持久化的事件流（P6）
或直接运行后粘贴需求。

需要环境变量（见 .env.example）：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL。
持久化（P6）由 PERSISTENCE_BACKEND 决定：
- sqlite（默认）→ checkpoint 落 CHECKPOINT_DB_PATH（默认 data/checkpoints.db），事件仅在内存；
- postgres      → checkpoint 与事件都落 DATABASE_URL（先 docker compose up -d）。

P7 起装配与 interrupt 循环抽到 `runtime/`（与 FastAPI 服务层共用），本文件只剩
「解析参数 + 打印 + 调 runtime」——**所有 print 及其相对顺序刻意保持不变**。
"""
from __future__ import annotations

import argparse
import sys
import uuid

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出，避免中文乱码
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from config.settings import Settings
from events.events import (
    RERUN_MARK,
    EventType,
    bind_thread,
    emit,
    emitter,
    format_event,
    replay_stream,
)
from persistence.checkpointer import PersistenceError, resolved_backend_label
from persistence.event_store import EventStoreError, PostgresEventStore
from runtime.assembly import build_runtime
from runtime.driver import run_task
from tools.command_runner import DockerSandboxError


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
    parser.add_argument(
        "--events",
        metavar="ID",
        help="回放指定会话的持久化事件流（需 PERSISTENCE_BACKEND=postgres）",
    )
    return parser.parse_args(argv)


def replay_events(thread_id: str) -> None:
    """从 PostgreSQL 读回一个会话的事件流，用与实时输出相同的格式打印。

    读侧复用 events.format_event，所以回放看到的就是当初终端上的样子。
    resume 后被 interrupt 恢复的节点会从头重跑、发出重复事件（step/node/type/message
    四项与首次完全相同，已探针实证），这里识别并标注「↻ 重跑」。
    """
    settings = Settings.from_env()
    if settings.persistence_backend != "postgres":
        print("错误：事件回放需要 PERSISTENCE_BACKEND=postgres。")
        print("  当前是 sqlite 后端 —— 事件只存在进程内存里，会话结束即丢弃、不落库。")
        print("  → 起库: docker compose up -d   然后设 PERSISTENCE_BACKEND=postgres")
        return

    try:
        with PostgresEventStore(settings.database_url) as store:
            events = store.load(thread_id)
    except EventStoreError as e:
        print(f"错误：读取事件失败。{e}")
        return

    if not events:
        print(f"会话 {thread_id} 没有事件记录（会话 ID 不对，或该会话没在 postgres 后端下跑过）。")
        return

    print(f"=== 会话 {thread_id} 事件回放（{len(events)} 条）===\n")
    for line, rerun in replay_stream(events):
        print(line + (RERUN_MARK if rerun else ""))


def ask_approval(payloads: list[dict]) -> str:
    """CLI 的审批交互：打印每个待批准问题，阻塞读一行输入。

    返回空串也行——`runtime/driver.py` 会把空值按拒绝归一化成 "no"。
    """
    for payload in payloads:
        print(f"\n[需要批准] {payload.get('question', '(无说明)')}")
    return input("  输入 yes 批准 / no 拒绝：").strip().lower()


def main() -> None:
    args = parse_args()

    # P6: 回放模式只读事件流，不需要 task / LLM key / 图 —— 在所有校验之前提前返回
    if args.events:
        replay_events(args.events)
        return

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

    # 实时打印事件
    emitter.add_listener(lambda e: print(format_event(e)))

    event_store = None
    try:
        # P7: 装配全部搬进 runtime/assembly.py（与 FastAPI 服务层共用同一份）。
        # 这里只保留 print —— 输出字节不变靠这条保证。
        with build_runtime(
            settings, thread_id=thread_id, workspace_root=settings.workspace_root
        ) as rt:
            event_store = rt.event_store

            # P5: run_command 执行宿主。SANDBOX_MODE=docker → 每会话一个长驻沙箱容器
            #（host workspace bind mount → /workspace，会话结束 rm -f）；local（默认）→
            # 没有 runner，rt.sandbox() 是空包 → 行为与 P0–P4 完全一致。
            # 刻意打印在容器**启动之前**：容器起不来时这行是用户唯一的线索。
            if rt.runner:
                print(
                    f"沙箱模式：run_command 将在 Docker 容器 {rt.runner.container_name} "
                    f"（镜像 {rt.runner.image}）内执行\n"
                )

            # P6: postgres 后端下落库事件；sqlite 后端保持 P0–P5 行为（事件只在进程内存）。
            if event_store is not None:
                print(f"持久化：{resolved_backend_label(settings)}")

            # P6: 给事件打「会话归属」戳。LangGraph 提交节点任务时会 copy_context()，
            # 所以这里 set 的值在 agent/tools/condense 节点、interrupt 恢复路径、以及
            # 嵌套的 specialist 子图里都可见。刻意不 reset —— 收尾事件也要带 thread_id，
            # 而一个进程只跑一个会话（服务端由每个 worker 线程各自绑定）。
            bind_thread(thread_id)

            # P4-2/P7: Human Approval 的挂起-恢复循环在 runtime/driver.py，与 API 共用。
            with rt.sandbox():
                if resume_mode:
                    # 恢复：先确认会话存在，再决定是否追加新指令
                    n_msgs = rt.existing_message_count()
                    if n_msgs == 0:
                        print(f"错误：会话 {thread_id} 不存在或已清理。")
                        return
                    print(f"\n=== 恢复会话 {thread_id}（已有 {n_msgs} 条消息）===")
                    if task:
                        print(f"追加指令: {task}")
                        graph_input = rt.append_input(task)
                    else:
                        graph_input = None
                else:
                    emit(EventType.AGENT_STARTED, agent="Supervisor", message="")
                    print(f"\n=== 任务 ===\n{task}\n")
                    print(f"会话 ID: {thread_id}")
                    graph_input = rt.initial_input(task)

                result, _paused = run_task(
                    rt.graph, graph_input, rt.config, on_interrupt=ask_approval
                )
    except DockerSandboxError as e:
        # docker 模式 daemon/镜像缺失只在 __enter__ 冒泡（run() 内已转 ERROR 文本回流 LLM）。
        # 给用户明确指引，不 traceback。
        print(f"\n错误：Docker 沙箱启动失败。{e}")
        return
    except (PersistenceError, EventStoreError) as e:
        # 同上：基础设施问题只报错不 traceback（P5 的处理方式，对称）
        print(f"\n错误：持久化初始化失败。{e}")
        return

    # P4-3-2 可观测：会话累计真实 token 消耗（汇总每次 LLM 调用的 usage）
    usage_events = [e for e in emitter.events if e.type is EventType.TOKEN_USAGE]
    if usage_events:
        prompt = sum((e.detail or {}).get("prompt_tokens", 0) for e in usage_events)
        completion = sum((e.detail or {}).get("completion_tokens", 0) for e in usage_events)
        print(
            f"\n=== Token 消耗 ===\nLLM 调用 {len(usage_events)} 次，"
            f"累计 输入 {prompt} / 输出 {completion} = {prompt + completion} tokens"
        )

    # P6: 事件持久化降级过就明确告知丢了多少 —— 否则用户永远不知道 events 表不全
    if event_store is not None and event_store.dropped_count:
        print(
            f"\n=== 事件持久化 ===\n已降级，本次共丢失 {event_store.dropped_count} 条事件"
            "（内存中仍在，但未落库）。",
        )

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
