# CodePilot

LangGraph 驱动的 Multi-Agent 软件工程运行时。P0 阶段：**单 Agent + ReAct + Tools + Workspace**。

> 架构基线来自 `../langgraph-agent-design/ARCHITECTURE.md` 与 `../openhands-agent-study/` 的
> OpenHands 源码分析。**OpenHands 只是参考对象，不是要复制**：我们提炼它的核心思想
> （ReAct 循环、工具契约、状态机、tool_call_id 配对），用 LangGraph 重新落地。

## 目录结构

```
codepilot/
├── main.py                 # CLI 入口
├── agent/
│   ├── state.py            # AgentState(TypedDict)：messages 为唯一真源
│   ├── core.py             # ReAct 内核：agent 节点 + tools 节点 + 条件路由
│   ├── graph.py            # 组装 StateGraph + MemorySaver
│   └── prompts.py          # system prompt
├── tools/
│   ├── filesystem.py       # list_files/read_file/write_file/edit_file/delete_file/search_code
│   ├── terminal.py         # run_command
│   └── registry.py         # 工具注册表
├── workspace/
│   └── manager.py          # 路径穿越守卫 + 受控文件操作
├── events/
│   └── events.py           # 事件系统（供 CLI / 未来 Web 消费）
├── config/
│   └── settings.py         # 环境变量配置
└── tests/                  # 单元测试 + FakeLLM 闭环测试
```

## 安装

```powershell
uv sync            # 依据 pyproject.toml 安装依赖到 .venv
```

## 配置（环境变量）

```powershell
$env:LLM_API_KEY  = "sk-..."
$env:LLM_BASE_URL = "https://api.deepseek.com/v1"   # DeepSeek 示例
$env:LLM_MODEL    = "deepseek-chat"
```

## 运行

```powershell
# 方式一：命令行参数
.venv\Scripts\python.exe main.py "创建一个 Python 快速排序程序，并创建单元测试"

# 方式二：交互式输入
.venv\Scripts\python.exe main.py
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

`tests/test_agent_loop.py` 用脚本化 FakeLLM **确定性**跑通完整 ReAct 闭环
（write_file → write_file → run_command → 完成），无需 API key / 网络。

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

- **P0**（当前）：单 Agent + ReAct + Tools + Workspace
- P1：Checkpoint / 持久化（MemorySaver → SqliteSaver）
- P2：Supervisor + 多 Agent + Subgraph（Analyst/Planner/Coder/Tester/Debugger/Reviewer）
- P3：Human Approval + interrupt
- P4：Error Recovery + Condense
- P5+：Docker Sandbox / PostgreSQL / FastAPI / React / Git
