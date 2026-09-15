"""P7: WebSocket 端点 —— **只订阅事件流**，不收动作（动作走 REST）。

一条连接 = 一个会话的事件订阅者。连上立刻发：

1. 一条 `kind=status`（当前状态 + 待批准的 payload）——新开的页面据此恢复按钮；
2. 整段 `kind=event` 回填（可带 `?since=<seq>` 跳过已收过的）；
3. 之后实时推。

推**结构化 Event**而不是 `format_event()` 的渲染结果：后者有损（AGENT_STARTED /
AGENT_COMPLETED 丢 message、tool 事件丢 `[agent]` 归属、CONDENSE 变多行字符串），
而 UI 要的恰恰是「哪个 agent 在干什么」。

## 两个必须记住的点

**1. `loop` 只能在 handler 内取。** `asyncio.get_running_loop()` 拿的是**当前**正在跑的
loop —— `TestClient` 把 app 跑在自己的 portal 线程、uvicorn 跑在它自己的 loop 上，
在 import / lifespan 里缓存一个 loop 会缓存到别人的，`call_soon_threadsafe` 便静默
投递到没人跑的 loop，事件永远不来且不报错。

**2. 不读 receive 就发现不了断开。** worker 线程只在有事件时才推东西；一个安静的会话
（比如正卡在几十秒的 LLM 调用上）两边都没动静，只靠 `send` 是察觉不到对端已经走了的
——handler 会抱着订阅者永远挂着。所以并发跑一个「读 receive 等断开」的任务，
谁先结束就取消另一个。

**3. 推失败绝不反噬会话。** `session._publish` 已经把订阅者回调包在 try 里了，
这里再兜一层是防 `send_json` 自己抛（对端半死）——可观测性不该杀死正在跑的任务。

**4. 收尾里不能 await。** 对端断开时 starlette 直接 `task.cancel()` 掉整个 handler，
取消异常正穿过 `_pump_until_closed`；此时在 `finally` 里 `await`（哪怕只是
`gather(..., return_exceptions=True)`）会用一个**丢掉取消消息的新 CancelledError**
顶掉原来那个。anyio 的 cancel scope 靠消息字符串认领自己的取消（`is_anyio_cancellation`），
认不出就不吞 → 异常穿出 starlette 的 `_run` → future 被标 cancelled →
`WebSocketTestSession.__exit__` 抛 `concurrent.futures.CancelledError`（真机踩过）。
所以收尾只 `cancel()` 不 await，异常由 `_drain` 取回。
"""
from __future__ import annotations

import asyncio
import sys

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from runtime.session import Session

router = APIRouter()

# 4000–4999 是应用自定义关闭码；用 4404 让前端能区分「会话没了」和「网络断了」。
WS_CLOSE_NO_SESSION = 4404
WS_CLOSE_NOT_READY = 4503


@router.websocket("/sessions/{thread_id}/ws")
async def session_events(websocket: WebSocket, thread_id: str, since: int = 0) -> None:
    registry = getattr(websocket.app.state, "registry", None)
    await websocket.accept()
    if registry is None:  # pragma: no cover  lifespan 没跑起来才会到这
        await _fail(websocket, "服务尚未就绪", WS_CLOSE_NOT_READY)
        return

    session: Session | None = registry.get(thread_id)
    if session is None:
        # 会话在服务重启后就没了（内存目录不持久化）——前端据此提示「重开会话」。
        await _fail(websocket, f"会话 {thread_id} 不存在", WS_CLOSE_NO_SESSION)
        return

    # ⚠️ 必须在 handler 内取 loop（见模块 docstring #1）
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def push(envelope: dict) -> None:
        """订阅者回调：从 worker 线程把信封投进 event loop。

        唯一线程安全的姿势就是 `call_soon_threadsafe`——直接 `queue.put_nowait` 会与
        loop 里的 `queue.get` 抢内部状态（asyncio.Queue 不是线程安全的）。
        """
        loop.call_soon_threadsafe(queue.put_nowait, envelope)

    status_env, backlog = session.subscribe(push)
    try:
        await websocket.send_json(status_env)
        for envelope in backlog:
            if envelope["seq"] >= since:
                await websocket.send_json(envelope)
        await _pump_until_closed(websocket, queue)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001  见模块 docstring #3
        # **必须留声**：信封里出现非 JSON 原生类型时，抛点在 starlette 的 send_json 里，
        # 悄无声息地吞掉就等于客户端一个字节都收不到、服务端也没有任何痕迹
        #（真机踩过：回填跑过任务的会话 → 状态信封里带图 state → 静默死连接）。
        print(
            f"警告: WS 推送中断（会话 {thread_id}）: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
    finally:
        session.unsubscribe(push)


async def _pump_until_closed(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """推事件，直到对端断开（或发送失败）。见模块 docstring #2。"""
    sender = asyncio.create_task(_send_forever(websocket, queue))
    watcher = asyncio.create_task(_wait_for_disconnect(websocket))
    for task in (sender, watcher):
        task.add_done_callback(_drain)

    try:
        done, _ = await asyncio.wait({sender, watcher}, return_when=asyncio.FIRST_COMPLETED)
        # `_send_forever` 是死循环，只有**抛异常**才会自己进 done —— 冒泡给 handler 去留声。
        # 判据用 done 成员而不是 gather 的结果：取消也走 gather，而取消是正常收尾。
        if sender in done:
            sender.result()
    finally:
        # ⚠️ 收尾里**绝不能 await**（见模块 docstring #4）。取消异常此刻正穿过本函数，
        # 在 finally 里 `await asyncio.gather(...)` 会让 CPython 拿一个新造的、**丢掉
        # 取消消息**的 CancelledError 顶掉原来那个，而 anyio 的 cancel scope 恰恰靠消息
        # 认领自己的取消。认不出 → 不吞 → 异常穿出 starlette 的 `_run` → 它把 future
        # 标成 cancelled → `WebSocketTestSession.__exit__` 抛 `CancelledError`。
        sender.cancel()
        watcher.cancel()


def _drain(task: asyncio.Task) -> None:
    """取回任务异常，免得 asyncio 在 GC 时打「exception never retrieved」。"""
    if not task.cancelled():
        task.exception()


async def _send_forever(websocket: WebSocket, queue: asyncio.Queue) -> None:
    while True:
        envelope = await queue.get()
        await websocket.send_json(envelope)


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    """读 receive 直到对端断开。**客户端从不发消息，这只是为了拿到断开事件。**"""
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return


async def _fail(websocket: WebSocket, message: str, code: int) -> None:
    await websocket.send_json({"kind": "error", "message": message})
    await websocket.close(code=code)


__all__ = ["WS_CLOSE_NO_SESSION", "WS_CLOSE_NOT_READY", "router"]
