# CodePilot

LangGraph 驱动的 Multi-Agent 软件工程运行时。P0 阶段：**单 Agent + ReAct + Tools + Workspace**。

> 架构基线来自 `../langgraph-agent-design/ARCHITECTURE.md` 与 `../openhands-agent-study/` 的
> OpenHands 源码分析。**OpenHands 只是参考对象，不是要复制**：我们提炼它的核心思想
> （ReAct 循环、工具契约、状态机、tool_call_id 配对），用 LangGraph 重新落地。
>
> 完整设计/演进见 `DESIGN.md`；项目记忆与工作约定见 `CLAUDE.md`。

当前阶段 **P6**：Supervisor + 6 个 Specialist 多 agent 编排 + Human Approval（interrupt）+
Error Recovery + Condense（消息数 + token 预算双守卫）+ 实时 token 消耗可观测 +
**Docker Sandbox**（run_command 执行隔离：默认本机 subprocess，`SANDBOX_MODE=docker` 时命令进 Docker 容器跑）+
**PostgreSQL 持久化**（checkpoint 与事件流落库，`--events <会话ID>` 回放；默认 sqlite 行为不变）。

## 目录结构

```
codepilot/
├── main.py                 # CLI 入口（UTF-8 + UUID thread + --resume/--events + 批准循环 + recursion_limit）
├── agent/
│   ├── state.py            # AgentState(TypedDict)：messages 为唯一真源
│   ├── core.py             # ReAct 内核：agent 节点 + tools 节点 + 条件路由
│   ├── graph.py            # 组装 StateGraph + MemorySaver
│   └── prompts.py          # system prompt
├── tools/
│   ├── filesystem.py       # list_files/read_file/write_file/edit_file/delete_file/search_code
│   ├── terminal.py         # run_command（宿主 = CommandRunner）
│   ├── command_runner.py   # P5：Local / DockerCommandRunner + DockerSandboxError
│   └── registry.py         # 工具注册表 + build_tools_subset
├── Dockerfile.sandbox      # P5 沙箱镜像（python:3.12-slim + pytest）
├── persistence/            # P6
│   ├── checkpointer.py     # build_checkpointer：SqliteSaver（默认）/ PostgresSaver
│   └── event_store.py      # PostgresEventStore：事件落库 + 回放读取
├── workspace/
│   └── manager.py          # 路径穿越守卫 + 受控文件操作
├── events/
│   └── events.py           # 事件系统（打戳 + format_event + replay_stream；供 CLI / 未来 Web 消费）
├── config/
│   └── settings.py         # 环境变量配置（含 context_limit / reserve_tokens / persistence_backend）
├── docker-compose.yml      # P6：PostgreSQL（docker compose up -d）
├── data/                   # checkpoints.db（sqlite 后端，已 gitignore）
└── tests/                  # 117 个 pytest：FakeLLM 闭环 + supervisor + condense + token + P5/P6 沙箱与持久化
```

## 安装

```powershell
uv sync            # 依据 pyproject.toml 安装依赖到 .venv
```

## 配置（环境变量）

```powershell
$env:LLM_API_KEY  = "sk-..."                          # 必填
$env:LLM_BASE_URL = "https://api.deepseek.com"        # DeepSeek 示例
$env:LLM_MODEL    = "deepseek-v4-flash"
# $env:MAX_ITERATIONS = 20                            # 防死循环
# $env:CONTEXT_LIMIT  = 1000000                       # token 预算（None=关闭 token 守卫，仅消息数触发）
# $env:RESERVE_TOKENS = 200000                        # 预算外 headroom（默认 context_limit//5）
# $env:SANDBOX_MODE   = "docker"                      # P5：run_command 执行隔离（默认 local=本机；docker=沙箱）
# $env:PERSISTENCE_BACKEND = "postgres"               # P6：持久化后端（默认 sqlite=本地文件；postgres=落库）
# $env:DATABASE_URL        = "postgresql://codepilot:codepilot@localhost:5432/codepilot"
```

### Docker Sandbox（P5，可选）

默认 `run_command` 在本机 subprocess 执行（行为与早期版本一致）。设 `SANDBOX_MODE=docker`
后，agent 的 shell/python/pytest 命令改在 **每会话一个的长驻 Docker 容器** 里跑
（host workspace 经 bind mount 挂成容器 `/workspace`，文件工具仍走 host、两侧双向可见；
容器内 pip 副作用随会话结束 `rm -f` 连根清）。先构建镜像：

```powershell
docker build -t codepilot-sandbox:py3.12 -f Dockerfile.sandbox .
$env:SANDBOX_MODE = "docker"
.venv\Scripts\python.exe main.py "创建一个 Python 快速排序程序，并创建单元测试"
```

### PostgreSQL 持久化（P6，可选）

默认是 sqlite（checkpoint 落 `data/checkpoints.db`，事件只在进程内存里、会话结束即丢弃，
行为与 P5 完全一致）。设 `PERSISTENCE_BACKEND=postgres` 后 **checkpoint 与事件流都落 PostgreSQL**
——这是 Web 期（P7）能回放一整段会话历史的地基：

```powershell
docker compose up -d                                   # 起库（postgres:16-alpine + 命名卷）
$env:PERSISTENCE_BACKEND = "postgres"
$env:DATABASE_URL = "postgresql://codepilot:codepilot@localhost:5432/codepilot"
.venv\Scripts\python.exe main.py "创建一个 Python 快速排序程序，并创建单元测试"

.venv\Scripts\python.exe main.py --events <会话ID>      # 回放该会话的完整事件流
docker compose down                                   # 停（保留数据）；down -v 连数据一起删
```

回放用的是与实时输出**同一套渲染**，所以看到的就是当初终端上的样子；`resume` 后被
interrupt 恢复的节点会从头重跑、发出重复事件，回放会标出来：

```
[Supervisor] 调用 tool: delegate
[Supervisor] 调用 tool: delegate   ↻ 重跑（interrupt 恢复后节点从头执行）
```

## 运行

```powershell
# 方式一：命令行参数
.venv\Scripts\python.exe main.py "创建一个 Python 快速排序程序，并创建单元测试"

# 指定会话 / 恢复上次中断的任务 / 回放已持久化的事件流
.venv\Scripts\python.exe main.py --thread my-task "需求"
.venv\Scripts\python.exe main.py --resume <会话ID>
.venv\Scripts\python.exe main.py --events <会话ID>          # 需 PERSISTENCE_BACKEND=postgres
```

运行时会实时打印 ReAct 轨迹：

```
[Coder] 开始任务
[Coder] 决定调用 2 个工具: write_file, write_file
[Coder] 调用 tool: write_file
    [Tool] write_file(path='main.py', ...)
[Coder] 调用 tool: run_command
    [Tool] run_command(command='pytest')
...
[Coder] 完成
```

## 测试

```powershell
.venv\Scripts\python.exe -m pytest -v
```

117 个 pytest 全过，覆盖：单 Agent ReAct 闭环（FakeLLM 确定性）、工具/workspace 守卫、
SQLite 持久化与 resume、supervisor 多 agent 编排（委派链/父子隔离/只读边界）、Human Approval
interrupt 挂起恢复、Error Recovery（子图异常兜底）、Condense（消息数 + token 预算双守卫、
最新 tool_call↔ToolMessage 配对完整、事件可观测）、Docker Sandbox（CommandRunner 注入链路 /
Docker CLI 参数与错误回流 / bind mount 双向可见 / 会话清理）、PostgreSQL 持久化（事件打戳与
`copy_context()` 语义 / Jsonb 与 SQL NULL / **连接必须 autocommit** / `record()` 永不抛异常与
降级 / 回放判重 / sqlite 默认分支行为不变）。

两组集成测试（真 Docker 容器 / 真 PostgreSQL）在 daemon 或库不可用时**模块级自动跳过**，
不影响常规 `pytest`：

```powershell
# PG 集成测试读的是 TEST_DATABASE_URL（刻意不用 DATABASE_URL，避免污染你在用的库）
$env:TEST_DATABASE_URL = "postgresql://codepilot:codepilot@localhost:5432/codepilot"
.venv\Scripts\python.exe -m pytest -q tests/test_postgres_live.py
```

## 设计要点（对应 OpenHands 思想）

| OpenHands | CodePilot |
|---|---|
| `Agent.step` + `classify_response` | `agent/core.py` 的条件路由 `route` |
| `ConversationState` + `EventLog` JSON | `AgentState`(TypedDict) + `messages`(add_messages) + MemorySaver |
| `ToolDefinition` + `register_tool`/`resolve_tool` | Pydantic args_schema + `registry` |
| `_execute_action_event` | 自定义 `tools` 节点（异常→ToolMessage 回流，不炸图） |
| `FileStore`/`LocalFileStore` | `workspace/manager.py` 路径守卫 |
| `litellm` 传输层 | `ChatOpenAI(base_url=...)` 供应商无关 |

**为什么没有复制 OpenHands**：舍弃了事件树分支（`parent_id`/`active_branch`）与文件级 JSON 持久化，
改用 LangGraph 原生能力（add_messages 配对、checkpointer、条件边）。控制流从手写 `while True`
变成图边，但工具契约、状态机语义、错误回流这些"核心思想"完整保留。

## 路线

- **P0**（✅ 完成）单 Agent + ReAct + Tools + Workspace
- **P1**（✅ 完成）Checkpoint / SQLite 持久化 + UUID thread + resume
- **P2**（✅ 完成）Supervisor + Coder/Tester/Reviewer（delegate + 子图隔离）
- **P3-1**（✅ 完成）Analyst + Planner（只读分析/规划）
- **P3-2**（✅ 完成）Debugger（诊断闭环，只读 + run_command）
- **P4-1**（✅ 完成）Error Recovery（子图异常兜底）
- **P4-2**（✅ 完成）Human Approval（interrupt：coder 委派前人工批准）
- **P4-3-1**（✅ 完成）Condense（长会话 messages 压缩）
- **P4-3-2**（✅ 完成）Token-aware Context Budget + 实时 token 消耗可观测
- P4-3 后续：Condense 调优（recent 巨型工具结果截断、会话管理）
- **P5**（✅ 完成）Docker Sandbox（run_command 执行隔离：CommandRunner + 会话级容器 + bind mount）
- **P6**（✅ 完成）PostgreSQL + Event Persistence（checkpoint 与事件落库 + `--events` 回放）
- P7+：FastAPI + SSE / React / Git / Eval
