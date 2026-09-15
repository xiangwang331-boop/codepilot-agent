"""P7: 会话驱动循环 —— interrupt（Human Approval）语义的唯一实现。

CLI 与 API 共用这一个函数：差别只在 `on_interrupt` 回调。
装配漂移只是「CLI 能跑、服务不能跑」；挂起语义漂移会让同一个审批在两条路径下
行为不同——所以这个循环不能各写一份。
"""
from __future__ import annotations

from typing import Any, Callable

from langgraph.types import Command


def run_task(
    graph: Any,
    graph_input: Any,
    config: dict,
    *,
    on_interrupt: Callable[[list[dict]], str | None],
) -> tuple[dict, bool]:
    """跑一个会话任务，直到结束或需要人类介入。

    `graph_input`
        新任务传初始 state dict；恢复传 `Command(resume=...)` —— 两条路径同一个函数，
        不需要分支（state 全在 checkpointer 里，与 graph 对象身份无关）。

    `on_interrupt(payloads) -> str | None`
        收到一批 interrupt 时回调，`payloads` 是各 `interrupt()` 传入的值（通常 1 个）。
        - 返回字符串 = **阻塞式交互**（CLI 的 `input()`），循环用 `Command(resume=...)` 继续；
        - 返回 `None`   = **让出线程**（API：记录待批准、置会话状态、worker 线程结束）。
          恢复时用新的线程重新调本函数，传 `Command(resume=...)` 即可。

    返回 `(最终 state, 是否因待批准而暂停)`。
    """
    result = graph.invoke(graph_input, config)
    while True:
        snap = graph.get_state(config)
        if not snap.next:
            break
        if not snap.interrupts:
            break  # 有未执行节点但不是 interrupt（保守退出，避免死循环）

        payloads = [(it.value or {}) for it in snap.interrupts]
        answer = on_interrupt(payloads)
        if answer is None:
            return result, True

        result = graph.invoke(Command(resume=normalize_answer(answer)), config)

    return result, False


def normalize_answer(answer: Any) -> str:
    """归一化 resume 值。

    ⚠️ `agent/supervisor.py` 的判定是 `if answer != "yes"`，所以 resume 值必须
    **精确等于字符串 "yes"**。`Command(resume=True)` 会静默变成「用户拒绝」
    （True != "yes"），整条委派被丢掉且不报错——API 层顺手传个 `{"approved": true}`
    就会踩到。这里把 bool 归一化掉，空串也按拒绝处理（与 CLI 的 `answer or "no"` 一致）。

    公开给 `runtime/session.py`：API 的「让出线程」路径要自己造 `Command(resume=...)`，
    必须是同一个归一化函数，否则 Web 端批准会静默变拒绝。
    """
    if answer is True:
        return "yes"
    if answer is False or answer is None:
        return "no"
    return str(answer) or "no"


__all__ = ["normalize_answer", "run_task"]
