# CodePilot

**一个会自己写代码、自己跑测试、失败了自己修的 Multi-Agent 软件工程运行时。**

给它一个需求，它把活干完：读现有代码 → 定计划 → 写实现 → 写测试 → 跑测试 →
**挂了就自己定位、自己改、再跑一遍**，直到全绿；只在改业务代码前停下来等你点一下批准。

和「让模型生成一段代码丢给你复制」的区别在于：它全程在**真实的 workspace** 里操作——
真的创建文件、真的执行 `pytest`、真的读报错再改。跑完你手上是一个能跑的目录，
不是一段待粘贴的文本。

基于 LangGraph，6 个专职 agent 分工协作，CLI 与 Web 界面共用同一套编排与挂起语义。

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![Tests](https://img.shields.io/badge/pytest-248%20tests-brightgreen)
![Web](https://img.shields.io/badge/vitest-234%20tests-brightgreen)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 它是怎么干活的

一个 Supervisor 负责决策，把任务拆给 6 个专职 agent；每个 agent 都是同一个 ReAct 内核，
只换 system prompt 和工具集。它们各自独立 state、独立迭代预算，只靠「任务字符串下去、
结果字符串上来」通信。

| Agent | 职责 | 工具权限 |
|---|---|---|
| `analyst` | 摸清现状：相关代码在哪、现状如何 | 只读 |
| `planner` | 定改动计划 | 只读 |
| `coder` | 写实现 | 读写 + 执行 |
| `debugger` | 定位根因（只诊断，不改文件） | 只读 + 执行 |
| `tester` | 写测试、跑测试 | 读写 + 执行 |
| `reviewer` | 只读审查 | 只读 |

一个需求的完整回路大致是：

```
需求 → analyst 摸现状 → planner 定计划 → 【批准】→ coder 写实现 → tester 跑测试
                                                                    │
                                            ┌───── 通过 ─────────────┤
                                            │                       │
                                            ▼                    失败 ↓
                                    reviewer 审查 ← 复验 ← coder 修复 ← debugger 定位
                                            │
                                            ▼
                                          完成
```

那条「失败 → 诊断 → 修复 → 复验」的回路是它自己走完的，不用你推着走。

**几点值得说明的设计：**

- **权限边界是代码写死的，不是靠 prompt 自觉**。只读角色（analyst / planner / reviewer）
  的工具列表里**根本没有**写文件的工具；debugger 能跑命令但改不了文件。越权不是「不允许」，
  是**工具不存在**。
- **改业务代码前会停下来问你**。基于 LangGraph `interrupt()` 挂起，你点批准才继续。
  这个挂起状态是落库的，服务重启后照样能接着批。
- **命令可以在容器里跑**。`SANDBOX_MODE=docker` 时每个会话一个长驻容器，workspace 双向
  bind mount，会话结束连根销毁；agent 装错的包、起的野进程污染不到你的主机。
- **每一步都留痕**。谁在干什么、真实 token 消耗、上下文何时被压缩、最后写了哪些文件，
  都是结构化事件；CLI 逐行打印，Web 界面渲染成按 agent 分组的时间线。
- **长会话不会爆上下文**。消息数 + token 估算双守卫触发压缩，把中间历史压成摘要保留头尾，
  且绝不劈开最新的「工具调用 ↔ 结果」配对。

---

## 环境要求

| 依赖 | 版本 | 是否必需 | 说明 |
|---|---|---|---|
| **Python** | 3.13+ | 必需 | 用到 3.13 的语法与标准库 |
| **一个 OpenAI 兼容的 API Key** | — | 必需 | DeepSeek / OpenAI / 其他兼容端点都行 |
| [uv](https://docs.astral.sh/uv/) | 0.9+ | 推荐 | 装依赖用；不想用 uv 也可以用 pip |
| **Node.js** | 20+ | 用 Web 界面才需要 | 构建前端；只用 CLI 可以不装 |
| **Docker** | 任意近期版本 | 可选 | 只有 `SANDBOX_MODE=docker` 需要 |
| **PostgreSQL** | 16 | 可选 | 只有 `PERSISTENCE_BACKEND=postgres` 需要；默认 sqlite 不用装 |

> **开 Docker 沙箱是强烈建议的**：不开的话 agent 执行的命令直接跑在你本机上。

---

## 安装

```bash
git clone <这个仓库>
cd CodePilot

# 后端依赖（二选一）
uv sync                              # 推荐：严格按 uv.lock，环境完全可复现
pip install -r requirements.txt      # 不用 uv 就用这个，版本同样已全部锁定

# 前端依赖（只有要用 Web 界面才需要）
cd web && npm ci && cd ..
```

依赖清单在 [`requirements.txt`](requirements.txt)（由 `uv.lock` 生成，含全部间接依赖、
版本锁定）。要加依赖请改 `pyproject.toml`，别直接改这两个文件。

---

## 配置

所有配置走**环境变量**。在项目根目录建一个 `.env` 就行——**启动时会自动读取，
不用手动 export**：

```bash
cp .env.example .env
```

然后至少填上 Key：

```ini
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.deepseek.com     # 留空走 OpenAI 官方端点
LLM_MODEL=deepseek-v4-flash
```

> 加载规则：优先读 `.env`，没有才读 `.env.example`（所以你不建 `.env` 时，
> 跑的就是 `.env.example` 里的默认值）；**已存在的环境变量不会被覆盖**，
> 所以临时改某个值直接 `export` 比改文件优先级更高。

完整的变量清单和逐项注释见 [`.env.example`](.env.example)，常用的这些：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_API_KEY` | 空 | **必填**，否则真实调用会失败 |
| `LLM_BASE_URL` | 空（OpenAI 官方） | 兼容端点地址 |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名。DeepSeek 是 `deepseek-v4-flash` / `deepseek-v4-pro`，**不带 `deepseek/` 前缀** |
| `WORKSPACE_ROOT` | `<项目根>/workspace_data` | agent 操作的文件根目录。指到你自己的项目上，它就改你的真实文件 |
| `MAX_ITERATIONS` | `20` | 单个 agent 的迭代上限（防死循环） |
| `SANDBOX_MODE` | `local` | `local` = 命令跑在本机；`docker` = 跑在沙箱容器 |
| `PERSISTENCE_BACKEND` | `sqlite` | `sqlite` = checkpoint 落本地文件、事件只在内存；`postgres` = 两者都落库 |
| `DATABASE_URL` | 空 | postgres 后端的连接串 |
| `LOG_LEVEL` / `LOG_DIR` | `INFO` / `<项目根>/data/logs` | 服务端日志；**同时**打在终端和文件里 |

---

## 运行

### 1. CLI（最快跑通）

```bash
uv run main.py "写一个 CLI 小工具，配上单元测试"
```

默认 sqlite 后端，不需要数据库。它会在 `WORKSPACE_ROOT` 下真实地建文件、跑测试，
遇到改业务代码的地方停下来问你 `yes` / `no`。

**让它在你已有的项目里干活**——把根目录指过去，CLI 下 agent 的 workspace
就是这个目录本身，它会直接读写你的真实文件：

```bash
WORKSPACE_ROOT=/path/to/your/project uv run main.py "给 CLI 加一个 --dry-run 参数，并补上测试"
```

> ⚠️ 这会真的改你的文件。先在 git 干净的分支上跑。

### 2. Web 界面

前端是构建产物、**不进 git**（`api/static/` 被忽略），所以新克隆下必须先构建一次，
否则 `GET /` 是 404：

```bash
cd web && npm run build && cd ..

uv run uvicorn api.app:create_app --factory --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

建会话 → 发需求 → 实时看事件流 → 遇到审批点按钮。

> 服务端每个会话有**自己独立**的 workspace `<WORKSPACE_ROOT>/<会话ID>`，会话之间互不干扰
> （与 CLI 直接用根目录本身的行为不同）。

### 3. 前端开发模式（改前端用这个）

后端和前端各起一个终端，Vite 带热更新：

```bash
uv run uvicorn api.app:create_app --factory --port 8000   # 终端 1（后端）
cd web && npm run dev                                     # 终端 2 → http://127.0.0.1:5173
```

Vite 把 `/sessions` 前缀（HTTP 与 WebSocket 同规则）代理到 8000，所以不需要 CORS 中间件。

### 4. 可选：Docker 沙箱

```bash
docker build -t codepilot-sandbox:py3.12 -f Dockerfile.sandbox .
```

然后在 `.env` 里设 `SANDBOX_MODE=docker`。需要 Docker Desktop 已启动。

### 5. 可选：PostgreSQL

默认 sqlite 下**事件不落库，重启即丢**；想留历史、想回放，就换成 postgres：

```bash
docker compose up -d
```

然后在 `.env` 里设：

```ini
PERSISTENCE_BACKEND=postgres
DATABASE_URL=postgresql://codepilot:codepilot@localhost:5432/codepilot
```

重启服务后，之前跑过的会话会从库里恢复出来，历史可回放、可接着续跑。

---

## 命令参考

```bash
uv run main.py "需求"                    # 新任务，自动生成会话 ID
uv run main.py --thread my-task "需求"    # 指定会话 ID
uv run main.py --resume <会话ID>          # 恢复上次中断的任务，可再追加一条指令
uv run main.py --events <会话ID>          # 回放会话的完整事件流（需 postgres 后端）
```

服务层的接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/sessions` | 建会话（独立 workspace 目录 + 自己的沙箱容器） |
| `GET` | `/sessions` | 列出全部会话：本进程活动过的 + 重启后从库里恢复的历史 |
| `POST` | `/sessions/{id}/messages` | 提交需求；会话忙时返回 `409` |
| `POST` | `/sessions/{id}/approval` | `{"approved": true}` 批准 / `false` 拒绝 |
| `DELETE` | `/sessions/{id}` | 关会话（停 worker + 删容器 + 连库一起删） |
| `WS` | `/sessions/{id}/ws?since=` | 订阅事件流 |

---

## 测试

```bash
uv run pytest -q            # 后端 248 个
cd web && npm test          # 前端 234 个（纯函数，不需要浏览器）
cd web && npm run typecheck # tsc -b
```

后端覆盖 ReAct 闭环、工具与路径守卫、resume、多 agent 编排、interrupt 挂起恢复、
上下文压缩与 token 预算、Docker 沙箱、PostgreSQL 落库与回放、REST/WS 契约。

**真 Docker / 真 PostgreSQL 的集成用例在 daemon 或数据库不可用时会自动跳过**，
不影响常规 `pytest`。本机没起这两样时是 `237 passed, 11 skipped`；
跑完整集成（真连 PG）需要额外设 `TEST_DATABASE_URL`（刻意与 `DATABASE_URL` 分开，
因为那些用例会真往库里写数据）：

```bash
TEST_DATABASE_URL=postgresql://codepilot:codepilot@localhost:5432/codepilot uv run pytest -q
```

前端刻意把逻辑全压进 `web/src/model/` 的纯函数层——折叠算法要覆盖的边界
（未闭合的委派块、一批两个委派、逐字重放、同名工具连调两次……）只有在没有浏览器的
情况下才断言得起；组件层只做渲染、不写测试。

---

## 架构

```text
                        浏览器（React 19 + Vite）
                              │  REST 做动作 / WebSocket 订阅事件
                 ┌────────────▼────────────┐
                 │   api/   FastAPI 服务层  │  create_app + lifespan
                 └────────────┬────────────┘
                              │
     ┌────────────────────────▼────────────────────────┐
     │  runtime/   装配 · 驱动 · 会话 · 目录            │
     │  build_runtime / run_task / Session / Catalog   │
     │  ← CLI 与 Web 共用同一套装配，也共用挂起语义     │
     └────────────────────────┬────────────────────────┘
                              │
     ┌────────────────────────▼────────────────────────┐
     │  agent/     LangGraph 编排                      │
     │  Supervisor ──delegate 工具──▶ 6 个 Specialist  │
     │  每个 specialist 是同一个 ReAct 内核，独立 state │
     └────────────────────────┬────────────────────────┘
                              │
     ┌────────────────────────▼────────────────────────┐
     │  tools/     7 个工具 + CommandRunner 可插拔宿主  │
     │  workspace/ 路径守卫（所有访问限制在根目录内）   │
     └────────────────────────┬────────────────────────┘
                              │
              workspace/<会话ID>/   ← agent 真实操作的文件
```

7 个工具 = 6 个文件操作（`list_files` / `read_file` / `write_file` / `edit_file` /
`delete_file` / `search_code`）+ 1 个 `run_command`。

```
├── main.py            CLI 入口
├── api/               FastAPI 服务层（app / routes / ws / schemas）
│                      static/ ← 前端构建产物落点，已 gitignore
├── web/               React Web UI 源码（api / model / hooks / components / styles）
├── runtime/           CLI 与 API 共享层：装配 · 驱动 · 会话 · 注册表 · 目录
├── agent/             LangGraph 编排：state / core / graph / supervisor / specialists / condense
├── tools/             工具实现 + CommandRunner（Local / Docker 两种宿主）
├── persistence/       checkpoint 与事件落库（sqlite / postgres）
├── workspace/         路径穿越守卫
├── events/            事件类型 · 打戳 · 渲染 · 回放
├── config/            环境变量解析 + 日志配置
├── tests/             后端测试
├── docker-compose.yml PostgreSQL（可选）
└── Dockerfile.sandbox 沙箱镜像（可选）
```

**几条贯穿全局的铁律：**

- **`AgentState.messages` 是唯一真源**，靠 `add_messages` 自动配对工具调用与结果；
  工具抛异常会被转成消息回流给 LLM 自纠，而不是炸掉整张图。
- **父子完全隔离**：Supervisor 与每个 specialist 各有独立 state 与迭代预算。
- **CLI 与 Web 共享的不只有装配，还有挂起语义**——装配漂移只是「一边能跑」，
  循环漂移会让同一个审批在两条路径下行为不同，所以那层只有一份实现。
- **加新角色只改注册表**：在 `agent/specialists.py` 的 `SPECIALISTS` 加一条即可，
  编排逻辑零改动。

---

## 路线图

- [x] **P0–P1** ReAct 内核 · 7 工具 · Workspace 路径守卫 · Checkpoint / resume
- [x] **P2–P3** Supervisor + 6 Specialist · 父子隔离 · 只读/写入权限边界
- [x] **P4** 错误恢复 · 人工审批（interrupt）· 长会话压缩 · token 预算
- [x] **P5** Docker 沙箱（命令执行隔离）
- [x] **P6** PostgreSQL：checkpoint 与事件流落库 · `--events` 回放
- [x] **P7** FastAPI + WebSocket 服务层（每会话独立 workspace + 容器）
- [x] **P8** React Web UI
- [x] **P9** 会话与事件的读路径（重启后重建会话列表与历史）
- [x] **P9+** 服务端日志（终端 + 文件双出口）
- [ ] **P10** Git Workflow（分支 / 提交 / diff 审查）
- [ ] **P11** Eval（多 agent 编排的效果评测）

---

## 已知限制

诚实地写在前面：

- **没有认证、没有多租户**，默认只监听 `127.0.0.1`，别直接暴露到公网。
- **沙箱不是对抗恶意代码的安全边界**：容器内仍是 root，未做 cap-drop / network none。
  威胁模型是「防 agent 的命令污染你的主机」，不是「跑不受信任的代码」。
- **假设同一时刻只有一个服务进程**：启动时会按 label 回收上个进程遗留的沙箱容器，
  多实例并存会互相删容器（可用 `SWEEP_SANDBOX_ON_START=0` 关掉）。
- **已知缺陷（未修）**：同一批里出现两个都需要批准的委派时，第二个会被静默丢弃——
  事件流会以一张收不了口的委派卡 + 一条根 `AgentFailed` 结束。
  测试里有一个用例**故意钉住**了这个错误行为，修好后它会变红。
- **压缩只压中间历史**：recent 窗口内单个巨型工具输出（比如一次刷屏的 pytest 输出）
  不做截断，极端情况下可能反复触发压缩却压不下去。
- **sqlite 后端下事件不落库**：重启后会话能列出来，但时间线是空的（界面有横幅明说）。
- Web 界面**没有客户端路由**：未知深链会 404，会话选择走 `/?session=<会话ID>`。

---

## 致谢

架构灵感来自 [OpenHands](https://github.com/All-Hands-AI/OpenHands) ——
但**它是参考对象，不是复制对象**：我们提炼它的核心思想（ReAct 循环、工具契约、
状态机、tool_call_id 配对），用 LangGraph 的原生能力重新落地（`add_messages` 配对、
checkpointer、条件边），刻意舍弃了它的事件树分支与文件级 JSON 持久化。

## License

[MIT](LICENSE)
