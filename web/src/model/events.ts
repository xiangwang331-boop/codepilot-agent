/**
 * 事件流 → 时间线块。**整个 UI 的核心纯函数。**
 *
 * ## 后端发射顺序（已对着 `tests/test_cli_output.py:191-229` 的逐字基线核实）
 *
 * 一次「委派 coder + 审批挂起 + 批准」的真实序列：
 *
 * ```
 *  0  AgentStarted     Supervisor  message=""                      session.py:427（仅首轮）
 *  1  AgentStep        Supervisor  "决定调用 1 个工具: delegate"     core.py:85
 *  2  ToolCallStarted  Supervisor  "delegate"  detail={args}       core.py:119
 *     ── interrupt() 在这里抛出 ────────────────────────────────── supervisor.py:84
 *     [人批准 → POST /approval → Command(resume="yes") → 新 worker 线程]
 *  3  ToolCallStarted  Supervisor  "delegate"  ← 与 seq 2 **逐字相同**（tools 节点整段重跑）
 *  4  AgentStarted     coder       message=""                      supervisor.py:113  ← 只此一次
 *  5  AgentStep        coder       "决定调用 1 个工具: write_file"  core.py:85
 *  6  ToolCallStarted  coder       "write_file" detail={args}      core.py:119
 *  7  ToolCallCompleted coder      "write_file(path='…')"          core.py:130（**无 detail**）
 *  8  AgentStep        coder       "给出最终回答"                   core.py:93
 *  9  AgentCompleted   coder       message=text[:80]               supervisor.py:123
 * 10  ToolCallCompleted Supervisor "delegate(specialist='coder'…)" core.py:130
 * 11  AgentStep        Supervisor  "给出最终回答"                   core.py:93
 * 12  AgentCompleted   Supervisor  message=""  step=null           session.py:460（图外）
 * ```
 *
 * ## 由这张表推出的四条算法约束（每一条都踩过或差点踩到）
 *
 * 1. **重跑的表现是「第二条 `ToolCallStarted(delegate)`」，不是「第二个子 AgentStarted」。**
 *    子 agent 的 `AgentStarted` 全程只出现一次（interrupt 在 `supervisor.py:84` 抛出，
 *    根本没走到 :113）。所以识别挂起重跑要靠**判重键命中**，不是靠「又一个 AgentStarted」。
 * 2. **块要跨审批保持打开**：seq 3 到达时块是开的且 `childRuns === 0` → 记 `attempts++`，
 *    复用同一个块，**绝不新开**（否则会留下一个永远闭合不了的幻影空块）。
 * 3. **父 agent 的工具调用不嵌套**：`core.py:114-146` 是 `for` 循环里「emit STARTED →
 *    invoke → emit COMPLETED」，所以同名工具的 started/completed 天然相邻，
 *    且同一时刻**至多一个委派块是开的**（specialist 没有 delegate 工具，不会嵌套）。
 * 4. **`delegate` 的 `ToolCallCompleted` 不等于子任务成功**：拒绝批准（`supervisor.py:91`）
 *    与未知 specialist（:79）都是 `return "ERROR: …"` **字符串**，`tool.invoke` 不抛，
 *    于是照样发 `ToolCallCompleted` 而**零个子事件**；子 agent 失败时
 *    （`supervisor.py:116-127`）也是先发子 `AgentFailed`、再发 delegate 的 Completed。
 *    → 判「没干活」要看 `childRuns === 0`，判「子任务失败」要看子 `AgentFailed`。
 *
 * ## 拿不到的东西（后端限制，不绕过）
 *
 * - **工具结果原文**：`core.py:130-134` 的 `TOOL_CALL_COMPLETED` 没传 `detail`，
 *   所以只显示工具名 + 参数（参数在 STARTED 的 `detail.args` 里有）。
 * - **子 agent 的完整报告**：只在 supervisor 的 `ToolMessage` 里，事件流里只有
 *   `text[:80]` 的截断。
 */
import type { AgentEvent, CondenseDetail, TokenUsageDetail, ToolCallStartedDetail } from "../api/types";
import { replayKey, type TimelineEvent } from "./replay";

export type BlockStatus = "open" | "completed" | "failed" | "not_executed";

/** 一次普通工具调用。`delegate` **不会**变成 ToolRow —— 它是 DelegationBlock。 */
export interface ToolRow {
  kind: "tool";
  id: string;
  agent: string;
  name: string;
  /** 来自 STARTED 的 `detail.args`；Completed/Failed 没有 detail。 */
  args: Record<string, unknown> | null;
  status: "running" | "done" | "failed";
  at: string;
  isRerun: boolean;
  /** 收口事件的原 message（`Completed` 是 `name(…)`，`Failed` 是 `name: err`）。 */
  message: string;
}

/** 不属于任何委派块的零散事件：根生命周期、AgentStep、Token、Condense、失败。 */
export interface SimpleBlock {
  kind: "lifecycle" | "step" | "token" | "condense" | "failure";
  id: string;
  item: TimelineEvent;
}

/** 一次委派 = 一个可折叠块，子 agent 的事件全在里面。 */
export interface DelegationBlock {
  kind: "delegation";
  id: string;
  /** 被委派的 specialist（小写）。 */
  specialist: string;
  task: string;
  /**
   * 这个块被「重新执行」过几次。审批挂起后 tools 节点重跑 → **2**。
   * 1 = 一次跑完，没有被 interrupt 打断。
   */
  attempts: number;
  state: BlockStatus;
  /** 块**首次**出现时是否就带着重跑标记（一般 false，重跑是后续事件）。 */
  isRerun: boolean;
  /** 子 agent 发了几次 `AgentStarted` —— 判「到底干没干活」的唯一可靠信号。 */
  childRuns: number;
  /** 子 agent 成功时的 `text[:80]`（`supervisor.py:123`）。 */
  summary: string | null;
  blocks: Block[];
  openedSeq: number;
  closedSeq: number | null;
  at: string;
}

export type Block = DelegationBlock | ToolRow | SimpleBlock;

// ---------------------------------------------------------------- 小工具

/** 从事件的 message 里抠出工具名。三种形态各不相同（见 `core.py:119/125/130/133/139/142`）。 */
export function eventToolName(event: AgentEvent): string | null {
  const m = event.message;
  switch (event.type) {
    case "ToolCallStarted":
      // message 就是裸工具名
      return m || null;
    case "ToolCallCompleted": {
      // `name(brief_args)`
      const i = m.indexOf("(");
      return i > 0 ? m.slice(0, i) : m || null;
    }
    case "ToolCallFailed": {
      // `name: 未知工具` / `name: TypeError: …`
      const i = m.indexOf(":");
      return (i > 0 ? m.slice(0, i) : m) || null;
    }
    default:
      return null;
  }
}

/** 工具参数的一行摘要。长值（整份文件 content）必须截断——否则界面会被撑爆。 */
export function summarizeArgs(args: Record<string, unknown> | null, max = 120): string {
  if (!args) return "";
  const parts: string[] = [];
  for (const [k, v] of Object.entries(args)) {
    let text: string;
    if (typeof v === "string") {
      text = v.replace(/\s+/g, " ").trim();
      if (text.length > 60) text = `${text.slice(0, 60)}…(${v.length} 字符)`;
    } else {
      text = JSON.stringify(v) ?? String(v);
    }
    parts.push(`${k}=${text}`);
  }
  const joined = parts.join(", ");
  return joined.length > max ? `${joined.slice(0, max)}…` : joined;
}

/** 该事件是不是子 agent 的「跑了」括号。 */
function isChildBracket(event: AgentEvent): boolean {
  return (
    event.type === "AgentStarted" ||
    event.type === "AgentCompleted" ||
    event.type === "AgentFailed"
  );
}

/** 递归找一个块里有没有失败（子块里的也算）。 */
export function hasFailure(blocks: Block[]): boolean {
  return blocks.some((b) => {
    if (b.kind === "failure") return true;
    if (b.kind === "tool") return b.status === "failed";
    if (b.kind === "delegation") return b.state === "failed" || hasFailure(b.blocks);
    return false;
  });
}

/** 一个块里涉及到的所有 agent（含后代），给过滤器用。 */
export function blockAgents(block: Block): string[] {
  switch (block.kind) {
    case "tool":
      return [block.agent];
    case "delegation":
      return [block.specialist, ...block.blocks.flatMap(blockAgents)];
    case "lifecycle":
    case "step":
    case "token":
    case "condense":
    case "failure":
      return [block.item.event.agent];
  }
}

function simpleKind(event: AgentEvent): SimpleBlock["kind"] {
  switch (event.type) {
    case "AgentStarted":
    case "AgentCompleted":
      return "lifecycle";
    case "AgentStep":
      return "step";
    case "TokenUsage":
      return "token";
    case "Condense":
      return "condense";
    case "AgentFailed":
      return "failure";
    default:
      return "step";
  }
}

// ---------------------------------------------------------------- 折叠

/**
 * 把**已按 seq 排序**的时间线事件折成块列表。
 *
 * 单次左到右扫描 + 一个「当前打开的委派块」指针。因为 `delegate` 不会嵌套
 * （specialist 的工具集里没有 delegate），深度恒为 1，所以用指针而不是栈。
 */
export function buildTimeline(items: TimelineEvent[]): Block[] {
  const root: Block[] = [];
  let group: DelegationBlock | null = null;
  let counter = 0;
  const nextId = (): string => `b${counter++}`;

  /**
   * 判重键 → 由它开出来的那个块。
   *
   * **只跟「当前打开的块」比较是不够的**：一批里有两个 `delegate` tool_call 时
   * （CLAUDE.md 关键坑 #40，`snap.interrupts` 会连续多轮出现），第三轮重跑会重新发出
   * **第一个** delegate 的 `ToolCallStarted` —— 而此时当前块早就是第二个的了，
   * 于是第一个委派会被重复开成第二张卡（一张卡变两张，内容还是错的）。
   *
   * 用键查表就从根上避开了这个问题：重跑事件与首次逐字相同（有后端契约测试钉死），
   * 于是它必然命中自己那张卡。
   */
  const openedByKey = new Map<string, DelegationBlock>();

  /** 当前该往哪个列表里塞东西：委派块打开且事件属于它 → 块内；否则根。 */
  const scopeFor = (event: AgentEvent): Block[] =>
    group !== null && event.agent === group.specialist ? group.blocks : root;

  /** 在某个列表里找最近的、同名且还开着的工具行。 */
  const takeOpenRow = (scope: Block[], name: string): ToolRow | null => {
    for (let i = scope.length - 1; i >= 0; i--) {
      const b = scope[i];
      if (b && b.kind === "tool" && b.name === name && b.status === "running") return b;
    }
    return null;
  };

  for (const item of items) {
    const event = item.event;
    const name = eventToolName(event);

    // ---- 1. delegate 开始 ----
    if (event.type === "ToolCallStarted" && name === "delegate") {
      const args = (event.detail as ToolCallStartedDetail | null)?.args ?? {};
      const specialist = String(args.specialist ?? "(未知)").toLowerCase();
      const task = String(args.task ?? "");

      const key: string | null = replayKey(event);
      const prior: DelegationBlock | undefined =
        key === null ? undefined : openedByKey.get(key);

      // 主判据：这条开块事件以前出现过（逐字相同）→ 它就是那张卡的重跑。
      // 兜底：`step` 为 null 时判重键不存在，退化成「同一个 specialist + 同一个 task、
      // 块还开着、子 agent 一次没跑」—— 正是「interrupt 在子 agent 启动前抛出」的形状。
      //
      // 两处**显式类型标注**不是风格问题：`reopen` 会在下面被赋回 `group`，
      // 而 `group` 又参与 `fallback` 的推断 —— 不标就撞 TS7022（循环推断）。
      const fallback: boolean =
        key === null &&
        group !== null &&
        group.specialist === specialist &&
        group.task === task &&
        group.state === "open" &&
        group.childRuns === 0;
      const reopen: DelegationBlock | null = prior ?? (fallback ? group : null);

      if (reopen !== null) {
        // 挂起恢复后 tools 节点**整段从头重跑**（`driver.py:48` 的
        // `Command(resume=…)`）—— 记一次 attempts，**不新开块**（否则一批两个
        // 委派时会重跑出多余的卡）。
        //
        // 重跑会把已执行过的子事件**原样再发一遍**（一批多个 tool_call 时尤其明显：
        // 第三个 interrupt 轮次会把第一个委派的子事件整段重放）。所以这里连内容一起
        // 清空，让本轮重新填 —— 否则卡片里会出现两份一模一样的 write_file 行。
        // 「跑过几轮」由 `attempts` 表达，不需要靠堆叠重复内容来表达。
        reopen.attempts += 1;
        reopen.state = "open";
        reopen.closedSeq = null;
        reopen.blocks = [];
        reopen.childRuns = 0;
        reopen.summary = null;
        group = reopen;
      } else {
        group = {
          kind: "delegation",
          id: nextId(),
          specialist,
          task,
          attempts: 1,
          state: "open",
          isRerun: item.isRerun,
          childRuns: 0,
          summary: null,
          blocks: [],
          openedSeq: item.seq,
          closedSeq: null,
          at: event.timestamp,
        };
        root.push(group);
        if (key !== null) openedByKey.set(key, group);
      }
      continue;
    }

    // ---- 2. delegate 结束 ----
    if (
      (event.type === "ToolCallCompleted" || event.type === "ToolCallFailed") &&
      name === "delegate"
    ) {
      if (group !== null) {
        if (group.childRuns === 0) {
          // 子 agent 一次都没启动：被拒批准（:91）或未知 specialist（:79）。
          // 这两者从事件流上**无法区分**，所以文案不写「被拒绝」。
          group.state = "not_executed";
        } else if (hasFailure(group.blocks)) {
          group.state = "failed";
        } else {
          group.state = "completed";
        }
        group.closedSeq = item.seq;
        group = null;
      }
      continue;
    }

    // ---- 3. 普通工具 ----
    if (event.type === "ToolCallStarted") {
      const scope = scopeFor(event);
      scope.push({
        kind: "tool",
        id: nextId(),
        agent: event.agent,
        name: name ?? "(未知工具)",
        args: (event.detail as ToolCallStartedDetail | null)?.args ?? null,
        status: "running",
        at: event.timestamp,
        isRerun: item.isRerun,
        message: "",
      });
      continue;
    }

    if (event.type === "ToolCallCompleted" || event.type === "ToolCallFailed") {
      const scope = scopeFor(event);
      // 同名工具在同一批里是**顺序**执行的，所以最近的同名开行就是它。
      const row = name === null ? null : takeOpenRow(scope, name);
      if (row !== null) {
        row.status = event.type === "ToolCallFailed" ? "failed" : "done";
        row.message = event.message;
      } else {
        // 兜底：没有配对的 Started（回填被截断、或事件流从中途开始）。
        // 宁可渲染一行孤儿，也不静默丢事件。
        scope.push({
          kind: "tool",
          id: nextId(),
          agent: event.agent,
          name: name ?? "(未知工具)",
          args: null,
          status: event.type === "ToolCallFailed" ? "failed" : "done",
          at: event.timestamp,
          isRerun: item.isRerun,
          message: event.message,
        });
      }
      continue;
    }

    // ---- 4. 子 agent 的生命周期括号：不单独成行，只更新块的计数/摘要 ----
    if (isChildBracket(event) && group !== null && event.agent === group.specialist) {
      if (event.type === "AgentStarted") {
        group.childRuns += 1;
      } else if (event.type === "AgentCompleted") {
        if (event.message) group.summary = event.message;
      } else {
        // 子 agent 失败：这一条**必须**渲染出来，否则用户看不到失败原因。
        group.blocks.push({ kind: "failure", id: nextId(), item });
      }
      continue;
    }

    // ---- 5. 其余（根生命周期 / step / token / condense）按 agent 归属 ----
    scopeFor(event).push({ kind: simpleKind(event), id: nextId(), item });
  }

  return root;
}

// ---------------------------------------------------------------- 过滤

export interface FilterState {
  /** `"all"` 或某个 agent 名（小写；根是 `"supervisor"`）。 */
  agent: string;
  /** TokenUsage 默认隐藏：一次真实会话 27 次 LLM 调用 = 27 行噪音。 */
  showTokenUsage: boolean;
  /** 只看问题：保留失败、以及含失败的委派块。 */
  onlyProblems: boolean;
}

export const DEFAULT_FILTERS: FilterState = {
  agent: "all",
  showTokenUsage: false,
  onlyProblems: false,
};

/** 一个块是否满足 agent 条件（自身或其任何后代命中即可）。 */
function matchesAgent(block: Block, agent: string): boolean {
  if (agent === "all") return true;
  return blockAgents(block).some((a) => a.toLowerCase() === agent);
}

/**
 * 过滤块列表。**委派块只要有一个后代命中就整体保留** —— 否则过滤会把子事件
 * 抽走、留一个空壳卡片。
 */
export function applyFilters(blocks: Block[], filters: FilterState): Block[] {
  const out: Block[] = [];
  for (const block of blocks) {
    if (!filters.showTokenUsage && block.kind === "token") continue;
    if (!matchesAgent(block, filters.agent)) continue;

    if (block.kind === "delegation") {
      const inner = applyFilters(block.blocks, filters);
      if (filters.onlyProblems) {
        if (block.state !== "failed" && block.state !== "open" && !hasFailure(inner)) continue;
      }
      out.push({ ...block, blocks: inner });
      continue;
    }

    if (filters.onlyProblems) {
      if (block.kind === "tool") {
        if (block.status !== "failed" && block.status !== "running") continue;
      } else if (block.kind !== "failure") {
        continue;
      }
    }
    out.push(block);
  }
  return out;
}

/** 时间线里出现过的 agent（小写、去重、保序），喂给过滤器的下拉。 */
export function collectAgents(blocks: Block[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  const walk = (list: Block[]): void => {
    for (const b of list) {
      for (const a of blockAgents(b)) {
        const key = a.toLowerCase();
        if (!seen.has(key)) {
          seen.add(key);
          out.push(key);
        }
      }
      if (b.kind === "delegation") walk(b.blocks);
    }
  };
  walk(blocks);
  return out;
}

/** 便利：给 condense 块读 detail。 */
export function condenseDetail(event: AgentEvent): CondenseDetail | null {
  return (event.detail as CondenseDetail | null) ?? null;
}

/** 便利：给 token 块读 detail。 */
export function tokenDetail(event: AgentEvent): TokenUsageDetail | null {
  return (event.detail as TokenUsageDetail | null) ?? null;
}
