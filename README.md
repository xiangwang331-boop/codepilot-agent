# CodePilot

LangGraph 驱动的 Multi-Agent 软件工程运行时。P0 阶段：**单 Agent + ReAct + Tools + Workspace**。

> 架构基线来自 `../langgraph-agent-design/ARCHITECTURE.md` 与 `../openhands-agent-study/` 的
> OpenHands 源码分析。**OpenHands 只是参考对象，不是要复制**：我们提炼它的核心思想
> （ReAct 循环、工具契约、状态机、tool_call_id 配对），用 LangGraph 重新落地。
>
> 完整设计/演进见 `DESIGN.md`；项目记忆与工作约定见 `CLAUDE.md`。

当前阶段 **P8**：Supervisor + 6 个 Specialist 多 agent 编排 + Human Approval（interrupt）+
Error Recovery + Condense（消息数 + token 预算双守卫）+ 实时 token 消耗可观测 +
Docker Sandbox（run_command 执行隔离）+ PostgreSQL 持久化（checkpoint 与事件流落库）+
**常驻服务层**（FastAPI REST + WebSocket：浏览器建会话、实时看多 agent 干活、遇到审批点按钮；
每会话一个独立 workspace 目录 + 一个自己的沙箱容器，由服务端接管生命周期）+
**React Web UI**（`web/`：委派时间线折叠、产出文件面板、令牌面板、事件过滤、`?session=` 深链、
深色默认主题；构建产物托管在 `api/static/`）。CLI 与 Web 共用同一套装配与挂起语义，
CLI 行为逐字不变。

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
│   └── events.py           # 事件系统（打戳 + format_event + replay_stream + per-session emitter）
├── runtime/                # P7：CLI 与 API 共享的装配/驱动/会话/注册表
│   ├── assembly.py         # build_runtime：ExitStack 持 ws/runner/saver/graph/store
│   ├── driver.py           # run_task：跑到结束或挂起（on_interrupt 决定阻塞还是让出线程）
│   ├── session.py          # Session：状态机 + worker 线程 + 审批槽 + 订阅者
│   └── registry.py         # SessionRegistry：create/get/list/teardown + 空闲回收
├── api/                    # P7：常驻服务层
│   ├── app.py              # create_app 工厂 + lifespan（预热 LLM / 建池 / 建表 / 清扫孤儿）
│   ├── routes.py           # REST 端点（建会话 / 发指令 / 审批 / 删会话 / 事件查询）
│   ├── ws.py               # WS 订阅端点（status + event 信封）
│   ├── schemas.py          # Pydantic 请求/响应模型
│   └── static/             # P8：React 构建产物落点（已 gitignore，先 npm run build）
├── web/                    # P8：React 前端（源码进 git，构建产物进 api/static）
│   ├── src/api/            # types.ts（JSON 契约镜像）/ client.ts（REST + 错误归一化）/ stream.ts（WS 客户端）
│   ├── src/model/          # **纯函数层**（折叠 / 判重 / token 聚合 / 文件重建 / 忙闲判定）—— vitest 只测这层
│   │   └── __fixtures__/   # 后端真实事件流夹具（由 tests/test_web_ui_contract.py 每次运行重写）
│   ├── src/hooks/ + components/   # 薄渲染层（React 19，零其他运行时依赖）
│   └── src/styles/         # 手写 CSS + 设计令牌（深色默认，[data-theme] 切浅色）
├── config/
│   └── settings.py         # 环境变量配置（含 context_limit / persistence_backend / api_host 等）
├── docker-compose.yml      # P6：PostgreSQL（docker compose up -d）
├── data/                   # checkpoints.db（sqlite 后端，已 gitignore）
└── tests/                  # 205 个 pytest：FakeLLM 闭环 + supervisor + condense + token + 沙箱/持久化 + API + UI 契约
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

# P7 服务层（只影响 `uvicorn` 起的服务；CLI 不读这些）
# $env:API_HOST = "127.0.0.1"                         # 默认只监听本机
# $env:API_PORT = "8000"
# $env:SESSION_IDLE_TIMEOUT = "1800"                  # 空闲会话回收秒数（running 态永不回收）
# $env:SWEEP_SANDBOX_ON_START = "1"                   # 默认开；设 0 关掉启动时对遗留沙箱容器的清扫
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

## 起服务（P7）

CLI 之外多了一个常驻服务：浏览器建会话、发需求、实时看多 agent 干活、遇到审批点按钮。
**默认 sqlite 模式即可跑**，不需要 PostgreSQL：

```powershell
.venv\Scripts\python.exe -m uvicorn api.app:create_app --factory --port 8000
# 浏览器打开 http://127.0.0.1:8000/  （P8 的 React UI；需先 npm run build，见下节）
```

> `--factory` 不能省：`create_app` 是**工厂函数**，刻意不写模块级 `app = create_app()`
> —— 那会让 import 时就跑 `Settings.from_env()`（有写 `os.environ` 的副作用），
> 且没法在测试里注入假 LLM / 假 runner。

### 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/sessions` | 建会话 → `201 {"thread_id","status"}`（每会话独立 workspace 目录 + 自己的沙箱容器） |
| `GET` | `/sessions` | 列出当前进程内所有会话 |
| `GET` | `/sessions/{id}` | 单个会话状态（`idle` / `running` / `awaiting_approval` / `closed`）+ 待批 payload |
| `POST` | `/sessions/{id}/messages` | 提交需求 → `202`；**会话忙时 `409`** |
| `POST` | `/sessions/{id}/approval` | `{"approved": true}` 批准 / `false` 拒绝 |
| `DELETE` | `/sessions/{id}` | 关会话 → `204`（停 worker + `docker rm -f` 容器 + 保留落盘产物） |
| `GET` | `/sessions/{id}/events?since=` | 拉历史事件（进程内环形缓冲，重启即空） |
| `WS` | `/sessions/{id}/ws?since=` | 订阅事件流 |

未知 id 一律 `404`；`task` 为空 `422`。WS 是**纯订阅**（动作全走 REST），连接即推一条
`status` 信封，之后每条事件一个信封——用 `kind` 区分，否则客户端分不清「会话忙」和「agent 事件」：

```json
{"kind":"status","status":"awaiting_approval","approval":[{"specialist":"coder","task":"..."}]}
{"kind":"event","seq":12,"event":{"type":"ToolCallStarted","agent":"coder","message":"write_file","detail":{"args":{"path":"main.py"}}}}
{"kind":"error","message":"..."}
```

`?since=N` 是**进程内序号**（不是 DB 游标），断线重连靠它补发；服务重启后序号从零开始，
需要跨重启的完整历史用 `PERSISTENCE_BACKEND=postgres` + `--events` 回放（见上）。

### 两个要注意的点

- **sqlite 模式下事件只在内存里**：不落库、服务重启即丢，`/events` 与 WS 的历史都只有本进程的。
  要持久化就设 `PERSISTENCE_BACKEND=postgres`（checkpoint 与事件流一起落库）。
- **启动清扫假设「同一时刻只有一个 CodePilot 服务进程」**：起服务时按 `label=codepilot.managed=1`
  回收上个进程遗留的沙箱容器（防异常退出后容器堆积）。**按 label 筛而不是按名字前缀**——后者会
  误删你自己起的同名容器。多进程部署前必须先关掉 `SWEEP_SANDBOX_ON_START`，否则会互相删容器。

## Web UI（P8）

React 19 + TypeScript + Vite，**后端一行不改**（P7 已经在推结构化事件，前端只是把它渲染出来）。
两个模式：

```powershell
# --- 生产态：构建产物流到 api/static，由 FastAPI 的 StaticFiles 托管（一个进程）---
cd web
npm ci                # 首次用 npm install（会生成 package-lock.json）
npm run build         # → ../api/static/{index.html, assets/*}
cd ..
.venv\Scripts\python.exe -m uvicorn api.app:create_app --factory --port 8000

# --- 开发态：Vite dev server + HMR，代理 /sessions 到后端（另开一个终端跑上面的 uvicorn）---
cd web
npm run dev           # http://127.0.0.1:5173/
```

> **构建产物不进 git**，所以新克隆下 `GET /` 是 404 —— 先 `npm run build`。
> 开发态之所以用 Vite 代理而不是 CORS：全仓没有 `CORSMiddleware`，加它就得改 `api/app.py`。

界面做的事：

- **委派时间线**——`ToolCallStarted(delegate)` 开一张卡（含 specialist + 任务），该 specialist 的
  子事件归入卡内，`ToolCallCompleted` 收口。**批准挂起时卡片保持打开态**，这本身就是「有个东西
  卡住了、等你点按钮」的可视化（原理见下）。恢复后子事件会整段重放，同一张卡标 `↻ 重跑` 并把
  `attempts` 加一，而不是平白多出一张卡。
- **产出文件面板**——从 `write_file`（带完整 `content`）/`edit_file`（`old_string`/`new_string`）
  的事件参数在客户端重建「本会话写过的文件」。**只能重建「写入」，看不到 agent 读过的文件原文。**
- **令牌面板 / 结果面板 / 事件过滤 / 多会话侧栏 / `?session=<id>` 深链**（可刷新可分享）。
- 深色默认，右上角切浅色（令牌落 `localStorage`）。

三个值得知道的点：

- **挂起在事件流里的样子是「一张收不了口的卡」**：`interrupt()` 在 `delegate` 的执行体内抛出，
  会**就地中断 tools 节点的循环**，所以那一次委派的 `ToolCallCompleted` 永远不会发出来。
  不变量是 **开块数 = 收口数 + 1**；这不是 bug，是前端判定「待批准」的唯一信号。
- **「一批里两个委派」会看到重放**：答完第一个中断后节点从头重跑，已批准的那次委派连同它的
  子事件会**逐字再来一遍**（第 11 条事件与第 2 条逐字相同），所以判重逻辑必须认得出它。
- **一轮里若出现两个都需要批准的委派，第二个会被静默丢弃**——这是 P7 遗留的后端缺陷
  （不与 P8 相关，未修），现象是事件流以一张收不了口的卡 + 一条根 `AgentFailed("状态 running")`
  结束。详见 `DESIGN.md` §7 的 P8 记录。

## 测试

```powershell
.venv\Scripts\python.exe -m pytest -v
```

205 个 pytest（其中 14 个在环境不满足时模块级自动跳过：9 个 PG 集成 + 5 个 Docker 沙箱集成），覆盖：单 Agent ReAct
闭环（FakeLLM 确定性）、工具/workspace 守卫、SQLite 持久化与 resume、supervisor 多 agent 编排
（委派链/父子隔离/只读边界）、Human Approval interrupt 挂起恢复、Error Recovery（子图异常兜底）、
Condense（消息数 + token 预算双守卫、最新 tool_call↔ToolMessage 配对完整、事件可观测）、
Docker Sandbox（CommandRunner 注入链路 / Docker CLI 参数与错误回流 / bind mount 双向可见 /
会话清理 / 启动清扫按 label 删孤儿且不误删无 label 容器）、PostgreSQL 持久化（事件打戳与
`copy_context()` 语义 / Jsonb 与 SQL NULL / **连接必须 autocommit** / `record()` 永不抛异常与
降级 / 回放判重 / sqlite 默认分支行为不变）、**服务层**（`tests/test_api.py`：状态机与 409 两态、
`{"approved": true}` → 精确 `"yes"`、WS 冒烟与重连、shutdown 停容器、per-session emitter 隔离）、
**UI 契约**（`tests/test_web_ui_contract.py`：字段名逐字对齐 `web/src/api/types.ts`、事件发射
顺序（委派嵌套/挂起重跑/拒绝闭合）、错误体形状、`write_file` 带完整 content、构建产物托管）。

前端单测是另一套（**不需要浏览器**，只测 `model/` + `api/` 的纯函数）：

```powershell
cd web
npm test              # vitest：折叠 / 判重 / token 聚合 / 文件重建 / 忙闲判定 / WS 状态机 / 错误归一化
npm run typecheck     # tsc -b
```

三组集成测试（真 Docker 容器 / 真 PostgreSQL / 真删容器做清扫）在 daemon 或库不可用时
**模块级自动跳过**，不影响常规 `pytest`：

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
- **P7**（✅ 完成）FastAPI + WebSocket 服务层（每会话独立 workspace + 容器、REST 动作 / WS 订阅、
  挂起审批点按钮、静态测试页、启动清扫孤儿容器）
- **P8**（✅ 完成）React Web UI（`web/`：委派时间线折叠 / 产出文件面板 / 令牌面板 / 事件过滤 /
  `?session=` 深链 / 深色默认主题；构建产物托管到 `api/static/`，**后端一行不改**）
- P9 起：Git / Eval
