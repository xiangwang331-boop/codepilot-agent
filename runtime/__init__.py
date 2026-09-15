"""P7: CLI 与 API 共享的运行时层。

- `assembly.py`：`build_runtime()` —— 一个会话的全部装配件（workspace / 沙箱 runner /
  checkpointer / 事件 store / supervisor 图 / config）。
- `driver.py`：`run_task()` —— 跑到结束或需要人类介入的循环（interrupt 语义的唯一实现）。
- `session.py`（P7-4）：`Session` —— 服务端会话的状态机 + worker 线程 + 审批让出/恢复
  + 订阅者（WS 信封协议）。
- `registry.py`（P7-4）：`SessionRegistry` —— 会话目录 + 空闲回收（`running` 永不回收）。
"""
