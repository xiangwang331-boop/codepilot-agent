"""P7: REST 端点 —— 只做「动作」，不推事件流（事件走 `api/ws.py`）。

HTTP 层不含任何状态：每个端点都是「翻译」——把 `Session` 状态机的返回值翻成状态码，
把请求体翻成状态机的入参。这样「会话忙」这种语义只有一处实现（`Session.begin()`
返回 False），HTTP 与未来的别的入口不会各判一套。

| 端点 | 语义 | 状态码 |
|---|---|---|
| `POST   /sessions` | 建会话（可选自定义 id） | 201 / 409（id 已占用） |
| `GET    /sessions` | 列出全部会话（**含重启后恢复的历史会话**） | 200 |
| `GET    /sessions/{id}` | 单个会话状态（历史会话**懒物化**） | 200 / 404 |
| `POST   /sessions/{id}/messages` | 发指令（新任务或追加） | 202 / 404 / **409 忙** |
| `POST   /sessions/{id}/approval` | 批准/拒绝挂起的委派 | 200 / 404 / **409 没在等审批** |
| `DELETE /sessions/{id}` | 销毁会话（收容器 + **连库一起删**） | 204 / 404 / **500 删库失败** |
| `GET    /sessions/{id}/events` | 事件回填（支持 `?since=`） | 200 / 404 |

**409 覆盖两个状态**：`running` 与 `awaiting_approval`。这不是保守起见——挂起中再
`invoke` 一次，LangGraph 会**静默吞掉**那个 interrupt（`next` 清空、待批准消失、
文件从未创建、无异常无日志，探针实证）。所以「忙」必须挡在这两个状态上，
绝不能用「worker 线程是否还活着」判断。

**P9：读路径接上了持久化。** 进程内存不再是会话目录的真源——`SessionRegistry.get()`
未命中内存时会从 `runtime/catalog.py` 的记录**懒物化**一个会话，所以重启后
`GET /sessions`、`GET /sessions/{id}`、WS 订阅、续跑、审批全部自动可用，
本文件的端点**一处签名都没改**（这是 P9 的设计目标：缺口在 registry 那一层补，
越往上改得越少）。唯一的例外是 `DELETE`：它现在要连库一起删，删库失败必须报错。
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status

from api.schemas import (
    ApprovalRequest,
    CreateSessionRequest,
    EventEnvelope,
    EventList,
    SessionInfo,
    SessionList,
    TaskRequest,
)
from config.logging_setup import get_logger
from runtime.catalog import CatalogError
from runtime.registry import SessionExistsError, SessionRegistry
from runtime.session import Session, SessionStatus

logger = get_logger(__name__)

router = APIRouter()


def get_registry(request: Request) -> SessionRegistry:
    """从 app.state 取注册表（lifespan 建的）。

    走 app.state 而不是模块级全局：测试用 `TestClient(create_app(...))` 起独立实例，
    两个 app 的注册表不会串（也用不着在测试里清全局）。
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover  lifespan 没跑起来才会到这（如测试直接调函数）
        raise HTTPException(status_code=503, detail="服务尚未就绪（lifespan 未完成）")
    return registry


Registry = Annotated[SessionRegistry, Depends(get_registry)]


def _require(registry: SessionRegistry, thread_id: str) -> Session:
    session = registry.get(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"会话 {thread_id} 不存在")
    return session


def _busy(session: Session, action: str) -> HTTPException:
    """409 的 detail 带上当前状态，前端据此决定显示什么控件（跑动中转圈 / 挂起中出按钮）。"""
    detail = {
        "status": session.status.value,
        "message": f"会话当前为 {session.status.value}，不能{action}",
    }
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


# ---------------------------------------------------------------- 会话目录


@router.post("/sessions", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
def create_session(
    registry: Registry,
    payload: Annotated[CreateSessionRequest | None, Body()] = None,
) -> SessionInfo:
    """新建会话。`thread_id` 留空自动生成。

    构造即装配（workspace / 图 / checkpointer），**但沙箱容器要等第一条指令才起**
    ——空会话不该白占一个容器（见 `Session._ensure_sandbox`）。
    """
    thread_id = payload.thread_id if payload else None
    try:
        session = registry.create(thread_id)
    except SessionExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    return SessionInfo(**session.snapshot())


@router.get("/sessions", response_model=SessionList)
def list_sessions(registry: Registry) -> SessionList:
    """列出全部会话：本进程活动过的在前，重启恢复的历史记录接在后。

    两段各自「最近活动在前」而不是混排——两段的时钟不可比（见 `registry.snapshots`）。
    `history_available=false` 时前端必须提示「有会话但没有事件流」。
    """
    return SessionList(
        sessions=[SessionInfo(**s) for s in registry.snapshots()],
        history_available=registry.history_available,
    )


@router.get("/sessions/{thread_id}", response_model=SessionInfo)
def get_session(thread_id: str, registry: Registry) -> SessionInfo:
    return SessionInfo(**_require(registry, thread_id).snapshot())


@router.delete("/sessions/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(thread_id: str, registry: Registry) -> Response:
    """销毁会话并移除（之后 GET 是 404）。容器一并 `rm -f`，**并连库一起删**。

    `running` 时也允许——这是用户的显式指令，堵住就没有别的办法收掉一个卡住的会话了
    （回收线程则相反：它只碰 `idle`/`awaiting_approval`/`interrupted`，因为用户并不知道它在动）。
    代价是正在跑的那条命令会以「容器没了」的 ERROR 收场，结果被丢弃。

    P9 起「移除」必须是**真删**：目录改成从库里读之后，只摘内存会让删掉的
    会话重启后复活——比不能删更糟。所以删库失败**不能**返回 204，那是在骗用户
    （他下次重启会看到它回来，而这一次的 204 让他以为删干净了）。
    """
    try:
        removed = registry.close(thread_id)
    except CatalogError as e:
        # 「删库失败要报 500 而不是 204」的另一半是**留下痕迹**：用户拿到 500 只说明
        # 「现在不对」，要知道「库里还残留着什么、下次重启会不会回来」得看这个栈。
        # 内存里那个会话此刻已经摘掉了、容器也关了 —— 这个不一致状态必须可查。
        logger.error(
            "删除会话 %s 失败：内存已移除，但持久化数据没删掉（重启后可能复活）",
            thread_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"会话 {thread_id} 已从内存移除，但持久化数据删除失败：{e}",
        ) from e
    if not removed:
        raise HTTPException(status_code=404, detail=f"会话 {thread_id} 不存在")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------- 动作


@router.post(
    "/sessions/{thread_id}/messages",
    response_model=SessionInfo,
    status_code=status.HTTP_202_ACCEPTED,
)
def post_message(thread_id: str, payload: TaskRequest, registry: Registry) -> SessionInfo:
    """发一条指令：会话还没有历史 → 全新任务；已有历史 → 追加一条（不重传初始 messages，
    否则 `add_messages` 会把它们追加成重复历史）。

    立刻返回 202：任务是**后台 worker 线程**跑的（`graph.invoke` 全程同步阻塞，
    放 event loop 会卡死整个服务），进展从 WS 看。
    """
    session = _require(registry, thread_id)
    if not session.begin(payload.task):
        raise _busy(session, "下发新指令")
    return SessionInfo(**session.snapshot())


@router.post("/sessions/{thread_id}/approval", response_model=SessionInfo)
def post_approval(
    thread_id: str, payload: ApprovalRequest, registry: Registry
) -> SessionInfo:
    """批准/拒绝挂起中的委派。

    `{"approved": true}` → `session.resume(True)` → `normalize_answer` → **精确的字符串
    `"yes"`**。这层必须归一化：`supervisor.py` 的判定是 `if answer != "yes"`，
    直接塞 bool 会**静默**变成「用户拒绝」（委派被丢、文件不写、不报错）。
    """
    session = _require(registry, thread_id)
    if not session.resume(payload.approved):
        raise _busy(session, "提交审批（当前没有待批准的委派）")
    return SessionInfo(**session.snapshot())


# ---------------------------------------------------------------- 事件回填


@router.get("/sessions/{thread_id}/events", response_model=EventList)
def get_events(
    thread_id: str,
    registry: Registry,
    since: Annotated[int, Query(ge=0, description="只返回 seq >= since 的事件")] = 0,
) -> EventList:
    """事件回填，支持 `?since=` 断线重连游标。

    **P9 起读的是持久化历史**（`PERSISTENCE_BACKEND=postgres`）：会话被懒物化时
    `catalog.load_events()` 已把整段历史按序灌进 `Session._events`，所以这里读到的
    下标即 seq、从 0 起连续，框架期完全没变。sqlite 后端的事件仍在进程内存里
    （重启即丢），此时 `list_sessions` 的 `history_available=false` 会让前端明说
    这件事——**不能假装一样**。

    `since` 的语义仍是 `seq >= since`（闭区间），所以断线重连要传
    `<最后收到的 seq> + 1`，传 lastSeq 会重复收到最后一条。
    """
    session = _require(registry, thread_id)
    events = [
        EventEnvelope(**env)
        for env in session.event_envelopes()
        if env["seq"] >= since
    ]
    return EventList(thread_id=thread_id, events=events)


__all__ = ["get_registry", "router"]
