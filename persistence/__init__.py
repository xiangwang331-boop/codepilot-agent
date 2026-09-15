"""P6 持久化层：checkpoint（会话状态）与事件流的落库实现。

- `checkpointer.build_checkpointer(settings)` —— 按 `PERSISTENCE_BACKEND` 选
  SqliteSaver（默认，P0–P5 行为）或 PostgresSaver。
- `event_store.PostgresEventStore` —— 事件落 PostgreSQL + 回放读取。

这一层只碰「存」，不碰「跑」：agent / tools / workspace 全部零改动。
"""
