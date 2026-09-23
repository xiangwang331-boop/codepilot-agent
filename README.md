# CodePilot

**一个会自己写代码、自己跑测试、失败了自己修的 Multi-Agent 软件工程运行时。**

给一个需求，它把活干完：读代码 → 定计划 → 写实现 → 写测试 → 跑测试 →
**测试挂了就自己定位、自己改、再跑**，直到全绿；只在关键动作（改业务代码）前停下来等你确认。

基于 LangGraph，6 个专职 agent 分工协作，全程在**真实的 workspace** 里操作——
真的建文件、真的执行 `pytest`、真的根据报错改代码。不是生成一段代码丢给你复制。

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![Tests](https://img.shields.io/badge/pytest-248%20tests-brightgreen)
![Web](https://img.shields.io/badge/vitest-234%20tests-brightgreen)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 一次自主循环长什么样

需求是「给这个项目加一个 `--dry-run` 参数，并补上测试」——**它得先读懂现有代码，才知道该改哪**：

```text
[Supervisor] 决定调用 1 个工具: delegate
[analyst] 开始任务
[analyst] 决定调用 1 个工具: search_code
    [Tool] search_code(query='argparse')
    [Tool] read_file(path='main.py')
[analyst] 给出最终回答：CLI 入口在 main.py 的 parse_args()，参数加了要同步 run_task 的调用点
[analyst] 完成

[Supervisor] 决定调用 1 个工具: delegate
[planner] 给出最终回答：改 2 个文件（main.py 加参数 + 传递，tests/test_cli_output.py 补用例）
[planner] 完成

[Supervisor] 决定调用 1 个工具: delegate
[需要批准] 是否批准委派 coder 执行任务？          ← 改业务代码前停下来问你
  输入 yes 批准 / no 拒绝：yes
[coder] 决定调用 1 个工具: edit_file
    [Tool] edit_file(path='main.py')
[coder] 完成

[tester] 决定调用 1 个工具: run_command
    [Tool] run_command(command='python -m pytest -q')
[tester] 给出最终回答：1 failed —— test_cli_output_matches_pre_p7_golden 断言失败，
        新增的 --dry-run 行打破了输出基线
[tester] 完成

[Supervisor] 决定调用 1 个工具: delegate
[debugger] 决定调用 1 个工具: read_file
    [Tool] read_file(path='tests/test_cli_output.py')
[debugger] 给出最终回答：基线与实际输出只差新增那一行；应更新基线而不是回退功能
[debugger] 完成

[需要批准] 是否批准委派 coder 执行任务？
  输入 yes 批准 / no 拒绝：yes
[coder] 决定调用 1 个工具: edit_file
    [Tool] edit_file(path='tests/test_cli_output.py')
[coder] 完成

[tester] 决定调用 1 个工具: run_command
    [Tool] run_command(command='python -m pytest -q')
[tester] 给出最终回答：248 passed —— 全绿
[tester] 完成

[reviewer] 给出最终回答：改动最小、测试覆盖了新增分支，无阻塞问题
[reviewer] 完成

=== Token 消耗 ===
LLM 调用 23 次，累计 输入 48210 / 输出 3915 = 52125 tokens

=== 结果 ===
已完成：新增 --dry-run 参数，输出基线已同步更新，pytest 全部通过。
```

注意这中间发生了什么：**测试失败了，于是它自己去定位、自己判断该改代码还是改测试、改完再跑一遍**。
这条「失败 → 诊断 → 修复 → 复验」的回路是它自己走完的，不是你推着走的。

> 上面是按真实事件流格式渲染的一次典型循环。跑起来看到的是实时逐行输出，
> Web 界面里则是按 agent 分组、可折叠的时间线。

## 为什么它敢自己跑

自主性不是靠 prompt 喊出来的，是这几层工程约束托出来的：

| | |
|---|---|
| **工具权限硬边界** | 每个 agent 拿到的工具集是**代码写死的**：只读角色（analyst/planner/reviewer）根本没有写文件的工具；debugger 能跑命令但不能改文件；tester 只碰 `test_*.py`。越权不是「靠 prompt 自觉」，是**工具不存在** |
| **人在环上** | 改业务代码前 `interrupt()` 挂起，等你点批准才继续。挂起状态可跨进程恢复——重启服务后照样能接着批 |
| **命令执行隔离** | `SANDBOX_MODE=docker` 时命令跑在每会话一个的长驻容器里，workspace 双向 bind mount，会话结束连根销毁。agent 装错的包、起的野进程污染不到你的主机 |
| **每一步都留痕** | 谁在干什么、真实 token 消耗、上下文何时被压缩、最终写了哪些文件，全部是结构化事件。**失败也看得见**——工具异常转成消息回流给 LLM 自纠，而不是静默崩掉 |
| **状态可恢复** | checkpoint + 事件流落 PostgreSQL。进程被杀掉，重启后会话还在、历史可回放、还能接着续跑 |
| **长会话不爆上下文** | 消息数 + token 估算双守卫触发压缩，把中间历史压成摘要、保留头尾；最新的 tool_call ↔ 结果配对绝不被劈开 |

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

6 个 specialist 各司其职：`analyst` 摸现状 · `planner` 定计划 · `coder` 写实现 ·
`debugger` 定位根因（只读）· `tester` 写测试并跑 · `reviewer` 只读审查。
加新角色只需在注册表加一条，编排逻辑零改动。

**几条贯穿全局的铁律：**

- **`AgentState.messages` 是唯一真源**，`add_messages` 负责 tool_call ↔ 结果配对；
  自定义 tools 节点把工具异常转成消息回流给 LLM 自纠，而不是炸掉整张图。
- **父子完全隔离**：supervisor 与每个 specialist 各有独立 state 与迭代预算，
  只靠「任务字符串下去、结果字符串上来」通信。
- **CLI 与 Web 共享的不只有装配，还有挂起语义**——装配漂移只是「一边能跑」，
  循环漂移会让同一个审批在两条路径下行为不同，所以那层只有一份实现。

## 快速开始

需要 **Python 3.13+**（[uv](https://docs.astral.sh/uv/) 管理依赖）、一个 OpenAI 兼容的 API Key。

```bash
uv sync

cat > .env <<'EOF'
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.deepseek.com     # 任何 OpenAI 兼容端点都行
LLM_MODEL=deepseek-v4-flash
EOF

# 默认 sqlite 后端，不需要数据库
uv run main.py "写一个 CLI 小工具，配上单元测试"
```

**让它在你已有的项目里干活**——把 `WORKSPACE_ROOT` 指过去就行（CLI 下 agent 的
workspace 就是这个目录本身，它会直接读写你的真实文件）：

```bash
WORKSPACE_ROOT=/path/to/your/project uv run main.py "给 CLI 加一个 --dry-run 参数，并补上测试"
```

> 服务端（Web）不一样：每个会话有**自己独立**的 workspace `<WORKSPACE_ROOT>/<会话ID>`，
> 会话之间互不干扰。

想用 Web 界面（**先构建前端**，产物不进 git）：

```bash
cd web && npm install && npm run build && cd ..
uv run uvicorn api.app:create_app --factory --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

> 前端开发态带 HMR：`cd web && npm run dev` → `http://127.0.0.1:5173`，
> Vite 把 `/sessions` 前缀（HTTP + WebSocket）代理到 8000，所以不需要 CORS 中间件。

**可选：Docker 沙箱**（强烈建议开着——否则 agent 的命令直接跑在你的机器上）

```bash
docker build -t codepilot-sandbox:py3.12 -f Dockerfile.sandbox .
# .env 里加：SANDBOX_MODE=docker
```

**可选：PostgreSQL**（默认 sqlite 下事件只在内存，重启即丢）

```bash
docker compose up -d
# .env 里加：
#   PERSISTENCE_BACKEND=postgres
#   DATABASE_URL=postgresql://codepilot:codepilot@localhost:5432/codepilot
```

完整环境变量清单见 [`.env.example`](.env.example)，每项都有注释。

## 常用命令

```bash
uv run main.py "需求"                   # 新任务，自动生成会话 ID
uv run main.py --thread my-task "需求"   # 指定会话 ID
uv run main.py --resume <会话ID>         # 恢复上次中断的任务，可再追加一条指令
uv run main.py --events <会话ID>         # 回放会话的完整事件流（需 postgres 后端）
```

服务层的 REST 端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/sessions` | 建会话（独立 workspace 目录 + 自己的沙箱容器） |
| `GET` | `/sessions` | 列出全部会话：本进程活动过的 + 重启后从库里恢复的历史 |
| `POST` | `/sessions/{id}/messages` | 提交需求；会话忙时 `409` |
| `POST` | `/sessions/{id}/approval` | `{"approved": true}` 批准 / `false` 拒绝 |
| `DELETE` | `/sessions/{id}` | 关会话（停 worker + 删容器 + 连库一起删） |
| `WS` | `/sessions/{id}/ws?since=` | 订阅事件流 |

## 测试

```bash
uv run pytest -q          # 248 个；环境不满足的集成用例模块级自动跳过
cd web && npm test        # 前端 234 个（只测纯函数，不需要浏览器）
cd web && npm run typecheck
```

后端覆盖 ReAct 闭环、工具与路径守卫、resume、多 agent 编排、interrupt 挂起恢复、
上下文压缩与 token 预算、Docker 沙箱、PostgreSQL 落库与回放、REST/WS 契约。
真 Docker / 真 PostgreSQL 的集成用例在 daemon 或库不可用时自动跳过，不影响常规 `pytest`。

前端刻意把逻辑全压进 `web/src/model/` 的纯函数层——折叠算法要覆盖的边界
（未闭合的委派块、一批两个委派、逐字重放、同名工具连调两次……）只有在没有浏览器时
才断言得起；组件层只做渲染、不写测试。

## 路线图

- [x] **P0–P1** ReAct 内核 · 7 工具 · Workspace 守卫 · Checkpoint / resume
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

## 已知限制

诚实地写在前面：

- **没有认证、没有多租户**，默认只监听 `127.0.0.1`，别直接暴露到公网。
- **沙箱不是对抗恶意代码的安全边界**：容器内仍是 root、未做 cap-drop / network none。
  威胁模型是「防 agent 的命令污染你的主机」，不是「跑不受信任的代码」。
- **假设同一时刻只有一个服务进程**：启动时会按 label 回收上个进程遗留的沙箱容器，
  多实例并存会互相删容器（可关 `SWEEP_SANDBOX_ON_START`）。
- **已知缺陷（未修）**：同一批里出现两个都需要批准的委派时，第二个会被静默丢弃——
  事件流会以一张收不了口的委派卡 + 一条根 `AgentFailed` 结束。
  `tests/test_web_ui_contract.py` 里有一个用例**故意钉住**了这个错误行为，修好后它会变红。
- **压缩只压中间历史**：recent 窗口内单个巨型工具输出（比如一次刷屏的 pytest 输出）
  不做截断，极端情况下可能反复触发压缩却压不下去。
- `sqlite` 后端下事件不落库，重启后会话能列出来但时间线是空的（界面有横幅明说）。

## 致谢

架构灵感来自 [OpenHands](https://github.com/All-Hands-AI/OpenHands) —— 但**它是参考对象，不是复制对象**：
我们提炼它的核心思想（ReAct 循环、工具契约、状态机、tool_call_id 配对），
用 LangGraph 原生能力重新落地（`add_messages` 配对、checkpointer、条件边），
刻意舍弃了它的事件树分支与文件级 JSON 持久化。

## License

[MIT](LICENSE)
