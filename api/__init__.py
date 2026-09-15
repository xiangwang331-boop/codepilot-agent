"""P7: FastAPI 服务层。

- `app.py`：`create_app()` —— 组装 + lifespan（装配共享池/事件 store/会话注册表，
  起回收线程，关停时逐个收干净）。**没有模块级 `app = create_app()`**：
  `Settings.from_env()` 有副作用（写 os.environ），且测试要注入假 LLM 工厂。
  起服务用 `uvicorn api.app:create_app --factory`。
- `routes.py`：REST 端点（动作）。
- `ws.py`：WS 端点（只订阅事件流）。
- `schemas.py`：请求/响应模型。

分层理由见 DESIGN.md 决策 #20/#21：**REST 做动作、WS 只推事件**，职责不重叠；
业务动作全部落到 `Session` 的状态机（`begin`/`resume`/`close`），HTTP 层只做
「翻译状态机返回值 → 状态码」和参数校验，不含任何状态。
"""
