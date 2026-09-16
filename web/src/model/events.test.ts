/**
 * 折叠算法测试。**分两层**：
 *
 * 1. **对真实夹具**（`__fixtures__/approval-{approved,rejected}.json`）—— 数据是后端
 *    真发出来的字节，所以这些用例证的是「算法对得上后端」，不是「算法对得上我推演的顺序」。
 *    后端哪天改了发射顺序，`tests/test_web_ui_contract.py` 先红，夹具跟着变，这里再红。
 * 2. **合成用例** —— 覆盖夹具造不出来的组合（同名工具连调、多 agent 并发、工具失败、
 *    未知 specialist、事件流从中途开始）。
 */
import { describe, expect, it } from "vitest";

import type { AgentEvent, EventEnvelope, EventType } from "../api/types";
import { APPROVED, BATCH, FIXTURE_TASK, REJECTED } from "./__fixtures__";
import {
  applyFilters,
  buildTimeline,
  collectAgents,
  DEFAULT_FILTERS,
  eventToolName,
  hasFailure,
  summarizeArgs,
  type Block,
  type DelegationBlock,
  type SimpleBlock,
  type ToolRow,
} from "./events";
import { markReruns, type TimelineEvent } from "./replay";

// ---------------------------------------------------------------- 造事件

let seq = 0;

/** 造一条事件。默认 `step=1, node="agent"` —— 图内事件。 */
function ev(
  type: EventType,
  agent: string,
  message = "",
  extra: Partial<AgentEvent> = {},
): AgentEvent {
  return {
    type,
    agent,
    message,
    detail: null,
    timestamp: "2026-01-01T00:00:00+00:00",
    thread_id: "t",
    step: 1,
    node: "agent",
    ...extra,
  };
}

/** 包成信封并重新编号（seq 必须单调，折叠与判重都依赖它）。 */
function env(...events: AgentEvent[]): EventEnvelope[] {
  return events.map((event) => ({ kind: "event" as const, seq: seq++, event }));
}

/** 每次用例前把 seq 归零，免得用例间的期望值互相影响。 */
function fresh<T>(fn: () => T): T {
  seq = 0;
  return fn();
}

/** 一步到位：造事件 → 判重 → 折叠。 */
function fold(events: AgentEvent[]): Block[] {
  return buildTimeline(markReruns(env(...events)));
}

const delegateStart = (specialist: string, task = "干活"): AgentEvent =>
  ev("ToolCallStarted", "Supervisor", "delegate", {
    step: 2,
    node: "tools",
    detail: { args: { specialist, task } },
  });

const delegateDone = (specialist: string, task = "干活"): AgentEvent =>
  ev("ToolCallCompleted", "Supervisor", `delegate(specialist='${specialist}', task='${task}')`, {
    step: 2,
    node: "tools",
  });

// ---------------------------------------------------------------- 真实夹具

describe("对真实夹具折叠（批准路径）", () => {
  const timeline = markReruns(APPROVED);
  const blocks = buildTimeline(timeline);
  const delegation = blocks.find((b) => b.kind === "delegation") as DelegationBlock;

  it("根层块数与顺序：生命周期 → step → 委派 → step → 生命周期", () => {
    expect(blocks.map((b) => b.kind)).toEqual([
      "lifecycle",
      "step",
      "delegation",
      "step",
      "lifecycle",
    ]);
  });

  it("委派块把 specialist 与 task 从 detail.args 里读出来了", () => {
    expect(delegation.specialist).toBe("coder");
    expect(delegation.task).toBe(FIXTURE_TASK);
  });

  it("挂起重跑记成 attempts=2，**不新开块**（否则会留下永远闭不了的幻影空块）", () => {
    expect(delegation.attempts).toBe(2);
    expect(blocks.filter((b) => b.kind === "delegation")).toHaveLength(1);
  });

  it("块以 completed 收口，closedSeq 指向 delegate 的收口事件", () => {
    expect(delegation.state).toBe("completed");
    expect(delegation.closedSeq).toBe(10);
    expect(delegation.childRuns).toBe(1);
  });

  it("子 agent 的 AgentCompleted 只贡献 summary，不单独成行", () => {
    expect(delegation.summary).toBe("快排已写入 main.py");
    expect(delegation.blocks.some((b) => b.kind === "lifecycle")).toBe(false);
  });

  it("块内 = [coder 的 step, write_file 行, coder 的 step]", () => {
    expect(delegation.blocks.map((b) => b.kind)).toEqual(["step", "tool", "step"]);
  });

  it("write_file 行配上了 Started/Completed，并留下 args（产出文件面板的依据）", () => {
    const row = delegation.blocks.find((b) => b.kind === "tool") as ToolRow;
    expect(row.name).toBe("write_file");
    expect(row.agent).toBe("coder");
    expect(row.status).toBe("done");
    expect(row.args).toEqual({ path: "main.py", content: "def quicksort(arr):\n    return sorted(arr)\n" });
  });

  it("根 AgentStarted 的 message 是空串，但块照样在（渲染层负责不显示空行）", () => {
    const first = blocks[0];
    expect(first?.kind).toBe("lifecycle");
    if (first?.kind !== "lifecycle") throw new Error("形状不对");
    expect(first.item.event.type).toBe("AgentStarted");
    expect(first.item.event.message).toBe("");
    // 图外事件（step=null）不参与判重
    expect(first.item.isRerun).toBe(false);
  });

  it("根 AgentCompleted 是图外事件（step=null），final 回答不在 message 里", () => {
    const last = blocks[blocks.length - 1];
    if (last?.kind !== "lifecycle") throw new Error("形状不对");
    expect(last.item.event.type).toBe("AgentCompleted");
    expect(last.item.event.step).toBeNull();
    // 最终回答走 SessionInfo.result，不走事件 —— 界面别指望从 message 拿
    expect(last.item.event.message).toBe("");
  });
});

describe("对真实夹具折叠（拒绝路径）", () => {
  const blocks = buildTimeline(markReruns(REJECTED));
  const delegation = blocks.find((b) => b.kind === "delegation") as DelegationBlock;

  it("委派块以 not_executed 收口（childRuns === 0）", () => {
    expect(delegation.state).toBe("not_executed");
    expect(delegation.childRuns).toBe(0);
  });

  it("attempts 仍是 2（挂起重跑照样发生），但块内一个子事件都没有", () => {
    expect(delegation.attempts).toBe(2);
    expect(delegation.blocks).toEqual([]);
  });

  it("收口事件是 Completed 而非 Failed —— 不能拿它反推子任务成功", () => {
    const closing = REJECTED.find(
      (e) => e.event.type === "ToolCallCompleted" && e.event.message.startsWith("delegate("),
    );
    expect(closing).toBeDefined();
    expect(REJECTED.some((e) => e.event.type === "ToolCallFailed")).toBe(false);
    expect(delegation.state).not.toBe("completed");
  });

  it("两条夹具的前 4 条事件逐字相同（「挂起」在数据里就是这么长的）", () => {
    const head = (list: EventEnvelope[]): string[] =>
      list.slice(0, 4).map((e) => JSON.stringify({ ...e.event, timestamp: "" }));
    expect(head(APPROVED)).toEqual(head(REJECTED));
  });
});

describe("对真实夹具折叠（一批两个委派）", () => {
  // 这份夹具是折叠算法最难对付的形状，两条难点叠在一起：
  //   1. 第二个委派的开块事件**收不了口**（`interrupt()` 在它的执行体里抛出，
  //      tools 节点的 `for` 循环就地中断，那一轮的 `C(b)` 永远发不出来）；
  //   2. 第二轮重跑把**第一个委派连同它的子事件整段重放**。
  // 所以裸事件顺序是 `S(a) C(a) S(b) | S(a) C(a) S(b) C(b)`（4 开 3 收）。
  const blocks = buildTimeline(markReruns(BATCH));
  const delegations = blocks.filter((b) => b.kind === "delegation") as DelegationBlock[];

  it("两个委派折成**两张卡**，不是三张（悬空的那个在重发时命中判重键复用）", () => {
    expect(delegations).toHaveLength(2);
    expect(delegations.map((d) => d.task)).toEqual(["写 a.py", "写 b.py"]);
  });

  it("两张卡都闭合在 completed —— 悬空的开块等到了自己迟到的收口", () => {
    expect(delegations.map((d) => d.state)).toEqual(["completed", "completed"]);
    expect(delegations.map((d) => d.attempts)).toEqual([2, 2]);
  });

  it("重放不会把子事件叠成两份：每张卡里只有一轮的内容", () => {
    // a.py 在外层确实跑了两次（两轮各一次），但卡里只留**最后一次重跑**填的内容
    // —— 4 条（step / tool / step / …）而不是 8 条。
    const a = delegations[0] as DelegationBlock;
    expect(a.childRuns).toBe(1);
    expect(a.blocks.filter((b) => b.kind === "tool")).toHaveLength(1);
    const writeRow = a.blocks.find((b) => b.kind === "tool") as ToolRow;
    expect(writeRow.args).toMatchObject({ path: "a.py" });
    expect(writeRow.status).toBe("done");
  });

  it("第二张卡的 childRuns 是 1 而不是 2 —— 它第一轮**根本没跑**（被 discard 的开块）", () => {
    const b = delegations[1] as DelegationBlock;
    expect(b.childRuns).toBe(1);
    expect(b.summary).toBe("b.py 写好了");
    expect((b.blocks.find((x) => x.kind === "tool") as ToolRow).args).toMatchObject({
      path: "b.py",
    });
  });

  it("块的 openseq 取首次出现、closedSeq 取最后那次收口", () => {
    const [a, b] = delegations as [DelegationBlock, DelegationBlock];
    const openSeqs = BATCH.filter(
      (e) => e.event.type === "ToolCallStarted" && e.event.message === "delegate",
    ).map((e) => e.seq);
    const closeSeqs = BATCH.filter(
      (e) => e.event.type === "ToolCallCompleted" && e.event.message.startsWith("delegate("),
    ).map((e) => e.seq);
    expect(openSeqs).toHaveLength(4);
    expect(closeSeqs).toHaveLength(3);
    expect(a.openedSeq).toBe(openSeqs[0]);
    expect(a.closedSeq).toBe(closeSeqs[1]); // 第二轮那次
    expect(b.openedSeq).toBe(openSeqs[1]); // 第一轮那条悬空的开块
    expect(b.closedSeq).toBe(closeSeqs[2]);
  });
});

// ---------------------------------------------------------------- 合成：工具配对

describe("工具行的配对", () => {
  it("同名工具在同一批里连调两次，各自配对（按顺序取最近的开行）", () => {
    const blocks = fresh(() =>
      fold([
        ev("ToolCallStarted", "coder", "read_file", { detail: { args: { path: "a.py" } } }),
        ev("ToolCallCompleted", "coder", "read_file(path='a.py')"),
        ev("ToolCallStarted", "coder", "read_file", { detail: { args: { path: "b.py" } } }),
        ev("ToolCallCompleted", "coder", "read_file(path='b.py')"),
      ]),
    );
    const rows = blocks.filter((b) => b.kind === "tool") as ToolRow[];
    expect(rows).toHaveLength(2);
    expect(rows.map((r) => r.status)).toEqual(["done", "done"]);
    expect(rows.map((r) => (r.args as { path: string }).path)).toEqual(["a.py", "b.py"]);
    expect(rows[0]?.message).toBe("read_file(path='a.py')");
    expect(rows[1]?.message).toBe("read_file(path='b.py')");
  });

  it("只有 Started 没有 Completed → 保持 running（这就是「进行中」的判定）", () => {
    const blocks = fresh(() =>
      fold([ev("ToolCallStarted", "coder", "run_command", { detail: { args: { command: "pytest" } } })]),
    );
    const row = blocks[0] as ToolRow;
    expect(row.status).toBe("running");
    expect(row.message).toBe("");
  });

  it("ToolCallFailed 把行标成 failed，并留下失败原文", () => {
    const blocks = fresh(() =>
      fold([
        ev("ToolCallStarted", "coder", "run_command", { detail: { args: { command: "pytest" } } }),
        ev("ToolCallFailed", "coder", "run_command: 未知工具"),
      ]),
    );
    const row = blocks[0] as ToolRow;
    expect(row.status).toBe("failed");
    expect(row.message).toBe("run_command: 未知工具");
    expect(hasFailure(blocks)).toBe(true);
  });

  it("孤儿 Completed（回填被截断）照样渲染一行，不静默丢事件", () => {
    const blocks = fresh(() => fold([ev("ToolCallCompleted", "coder", "write_file(path='x.py')")]));
    const row = blocks[0] as ToolRow;
    expect(row.name).toBe("write_file");
    expect(row.status).toBe("done");
    expect(row.args).toBeNull();
  });

  it("三种 message 形态都能抠出工具名", () => {
    expect(eventToolName(ev("ToolCallStarted", "a", "write_file"))).toBe("write_file");
    expect(eventToolName(ev("ToolCallCompleted", "a", "write_file(path='x')"))).toBe("write_file");
    expect(eventToolName(ev("ToolCallFailed", "a", "run_command: TypeError: x"))).toBe("run_command");
    expect(eventToolName(ev("AgentStep", "a", "决定调用 1 个工具: delegate"))).toBeNull();
    // 没有括号/冒号的畸形 message 原样返回，不返回空
    expect(eventToolName(ev("ToolCallCompleted", "a", "weird"))).toBe("weird");
  });
});

// ---------------------------------------------------------------- 合成：委派

describe("委派分组", () => {
  it("正常闭合：子事件全在块内，收口后 group 清空", () => {
    const blocks = fresh(() =>
      fold([
        delegateStart("coder", "写快排"),
        ev("AgentStarted", "coder", "", { step: 2, node: "tools" }),
        ev("ToolCallStarted", "coder", "write_file", {
          step: 2,
          node: "tools",
          detail: { args: { path: "m.py", content: "x" } },
        }),
        ev("ToolCallCompleted", "coder", "write_file(path='m.py')", { step: 2, node: "tools" }),
        ev("AgentCompleted", "coder", "写好了", { step: 2, node: "tools" }),
        delegateDone("coder", "写快排"),
      ]),
    );
    expect(blocks).toHaveLength(1);
    const d = blocks[0] as DelegationBlock;
    expect(d.state).toBe("completed");
    expect(d.attempts).toBe(1);
    expect(d.childRuns).toBe(1);
    expect(d.summary).toBe("写好了");
    expect(d.blocks.map((b) => b.kind)).toEqual(["tool"]);
  });

  it("挂起未闭合：只发了 Started 的块保持 open（= 待批准的可视化）", () => {
    const blocks = fresh(() => fold([delegateStart("coder", "写快排")]));
    const d = blocks[0] as DelegationBlock;
    expect(d.state).toBe("open");
    expect(d.closedSeq).toBeNull();
    expect(d.childRuns).toBe(0);
  });

  it("子 agent 失败：块内出现 failure 行，块判 failed", () => {
    const blocks = fresh(() =>
      fold([
        delegateStart("coder", "写快排"),
        ev("AgentStarted", "coder", "", { step: 2, node: "tools" }),
        ev("AgentFailed", "coder", "LLM 调用失败: 429", { step: 2, node: "tools" }),
        delegateDone("coder", "写快排"),
      ]),
    );
    const d = blocks[0] as DelegationBlock;
    expect(d.state).toBe("failed");
    expect(d.blocks.map((b) => b.kind)).toEqual(["failure"]);
    expect(hasFailure(blocks)).toBe(true);
  });

  it("子 agent 失败时 summary 留空（AgentFailed 不是 AgentCompleted）", () => {
    const blocks = fresh(() =>
      fold([
        delegateStart("coder", "x"),
        ev("AgentStarted", "coder", "", { step: 2, node: "tools" }),
        ev("AgentFailed", "coder", "炸了", { step: 2, node: "tools" }),
        delegateDone("coder", "x"),
      ]),
    );
    expect((blocks[0] as DelegationBlock).summary).toBeNull();
  });

  it("子 AgentCompleted 的 message 是空串时不留空 summary", () => {
    const blocks = fresh(() =>
      fold([
        delegateStart("coder", "x"),
        ev("AgentStarted", "coder", "", { step: 2, node: "tools" }),
        ev("AgentCompleted", "coder", "", { step: 2, node: "tools" }),
        delegateDone("coder", "x"),
      ]),
    );
    expect((blocks[0] as DelegationBlock).summary).toBeNull();
  });

  it("两次**不同** specialist 的委派 → 两个独立的块，且第二个的 childRuns 归零", () => {
    const blocks = fresh(() =>
      fold([
        delegateStart("coder", "写"),
        ev("AgentStarted", "coder", "", { step: 2, node: "tools" }),
        ev("AgentCompleted", "coder", "好了", { step: 2, node: "tools" }),
        delegateDone("coder", "写"),
        delegateStart("tester", "测"),
        ev("AgentStarted", "tester", "", { step: 4, node: "tools" }),
        ev("AgentCompleted", "tester", "全绿", { step: 4, node: "tools" }),
        delegateDone("tester", "测"),
      ]),
    );
    expect(blocks).toHaveLength(2);
    expect(blocks.map((b) => (b as DelegationBlock).specialist)).toEqual(["coder", "tester"]);
    expect(blocks.map((b) => (b as DelegationBlock).state)).toEqual(["completed", "completed"]);
    expect((blocks[1] as DelegationBlock).summary).toBe("全绿");
  });

  it("同一 specialist 但**任务不同** → 新开一块，不误判成重跑", () => {
    const blocks = fresh(() =>
      fold([
        delegateStart("coder", "第一件事"),
        ev("AgentStarted", "coder", "", { step: 2, node: "tools" }),
        ev("AgentCompleted", "coder", "done1", { step: 2, node: "tools" }),
        delegateDone("coder", "第一件事"),
        delegateStart("coder", "第二件事"),
        ev("AgentStarted", "coder", "", { step: 4, node: "tools" }),
        ev("AgentCompleted", "coder", "done2", { step: 4, node: "tools" }),
        delegateDone("coder", "第二件事"),
      ]),
    );
    expect(blocks).toHaveLength(2);
    expect(blocks.map((b) => (b as DelegationBlock).attempts)).toEqual([1, 1]);
  });

  it("块已闭合后又来一条**逐字相同**的 delegate Started（重跑）→ 复用成 attempts++", () => {
    // 这条对应「多 interrupt 轮次」：答完一个 tool_call 后节点重跑，再遇到下一个。
    const blocks = fresh(() =>
      fold([
        delegateStart("coder", "同一件事"),
        delegateDone("coder", "同一件事"),
        delegateStart("coder", "同一件事"),
        ev("AgentStarted", "coder", "", { step: 4, node: "tools" }),
        ev("AgentCompleted", "coder", "好了", { step: 4, node: "tools" }),
        delegateDone("coder", "同一件事"),
      ]),
    );
    expect(blocks).toHaveLength(1);
    const d = blocks[0] as DelegationBlock;
    expect(d.attempts).toBe(2);
    expect(d.state).toBe("completed");
    expect(d.childRuns).toBe(1);
  });

  it("未知 specialist 从事件流上与「被拒」不可区分 —— 都判 not_executed", () => {
    const blocks = fresh(() => fold([delegateStart("没人", "x"), delegateDone("没人", "x")]));
    expect((blocks[0] as DelegationBlock).state).toBe("not_executed");
  });

  it("specialist 名大小写归一化（后端做了 lower，前端跟着做以免对不上 agent 名）", () => {
    const blocks = fresh(() =>
      fold([
        delegateStart("Coder", "x"),
        ev("AgentStarted", "coder", "", { step: 2, node: "tools" }),
        ev("AgentCompleted", "coder", "ok", { step: 2, node: "tools" }),
        delegateDone("Coder", "x"),
      ]),
    );
    const d = blocks[0] as DelegationBlock;
    expect(d.specialist).toBe("coder");
    // 归一化对了，子事件才认得出归属
    expect(d.childRuns).toBe(1);
    expect(d.state).toBe("completed");
  });

  it("父 agent 的 step 事件不会被塞进块里（按 agent 名归属，不是靠 step 区间）", () => {
    const blocks = fresh(() =>
      fold([
        delegateStart("coder", "x"),
        // 父 agent 在委派**进行中**发的 step：它的 `step=2/node="tools"` 与子图的
        // 位置**撞在一起**，所以归属只能按 agent 名判，不能按 step 区间判。
        ev("AgentStep", "Supervisor", "决定调用 1 个工具: delegate", { step: 2, node: "tools" }),
        ev("AgentStep", "Supervisor", "给出最终回答", { step: 4, node: "agent" }),
        delegateDone("coder", "x"),
      ]),
    );
    // 3 条：委派块 + 两条**落在根上**的 Supervisor step
    expect(blocks.map((b) => b.kind)).toEqual(["delegation", "step", "step"]);
    const d = blocks[0] as DelegationBlock;
    expect(d.blocks).toEqual([]);
    expect((blocks[1] as SimpleBlock).item.event.agent).toBe("Supervisor");
  });

  it("details 缺失的 delegate Started 也能开块（specialist 退化成占位符）", () => {
    const blocks = fresh(() =>
      fold([ev("ToolCallStarted", "Supervisor", "delegate", { step: 2, node: "tools" })]),
    );
    const d = blocks[0] as DelegationBlock;
    expect(d.specialist).toBe("(未知)");
    expect(d.task).toBe("");
  });
});

// ---------------------------------------------------------------- 过滤

describe("过滤", () => {
  const blocks = fresh(() =>
    fold([
      ev("AgentStarted", "Supervisor", "", { step: null, node: null }),
      delegateStart("coder", "写"),
      ev("AgentStarted", "coder", "", { step: 2, node: "tools" }),
      ev("ToolCallStarted", "coder", "write_file", {
        step: 2,
        node: "tools",
        detail: { args: { path: "m.py", content: "x" } },
      }),
      ev("ToolCallFailed", "coder", "write_file: 磁盘满了", { step: 2, node: "tools" }),
      ev("AgentCompleted", "coder", "好了", { step: 2, node: "tools" }),
      delegateDone("coder", "写"),
      ev("TokenUsage", "Supervisor", "消耗 100 tokens", {
        step: 3,
        node: "agent",
        detail: { prompt_tokens: 80, completion_tokens: 20, total_tokens: 100 },
      }),
    ]),
  );

  it("TokenUsage 默认隐藏（一次真实会话 27 条 = 27 行噪音）", () => {
    const visible = applyFilters(blocks, DEFAULT_FILTERS);
    expect(visible.some((b) => b.kind === "token")).toBe(false);
  });

  it("打开开关后 TokenUsage 出现", () => {
    const visible = applyFilters(blocks, { ...DEFAULT_FILTERS, showTokenUsage: true });
    expect(visible.some((b) => b.kind === "token")).toBe(true);
  });

  it("按 agent 过滤：**委派块只要有后代命中就整体保留**（不抽成空壳）", () => {
    const visible = applyFilters(blocks, { ...DEFAULT_FILTERS, agent: "coder" });
    expect(visible.map((b) => b.kind)).toEqual(["delegation"]);
    const d = visible[0] as DelegationBlock;
    expect(d.blocks.length).toBeGreaterThan(0);
    // Supervisor 的根生命周期被滤掉了
    expect(visible.some((b) => b.kind === "lifecycle")).toBe(false);
  });

  it("按 agent 过滤：块头的 specialist 自己也算命中", () => {
    const visible = applyFilters(blocks, { ...DEFAULT_FILTERS, agent: "coder" });
    expect(visible).toHaveLength(1);
  });

  it("过滤器 agent 名大小写不敏感（根是 \"Supervisor\"，过滤值是 \"supervisor\"）", () => {
    const visible = applyFilters(blocks, { ...DEFAULT_FILTERS, agent: "supervisor" });
    expect(visible.some((b) => b.kind === "lifecycle")).toBe(true);
  });

  it("onlyProblems：保留失败工具行与含失败的委派块", () => {
    const visible = applyFilters(blocks, { ...DEFAULT_FILTERS, onlyProblems: true });
    expect(visible.map((b) => b.kind)).toEqual(["delegation"]);
    const d = visible[0] as DelegationBlock;
    expect(d.blocks.map((b) => b.kind)).toEqual(["tool"]);
    expect((d.blocks[0] as ToolRow).status).toBe("failed");
  });

  it("onlyProblems 连 open 的块一起保留（进行中也算「要看」）", () => {
    const open = fresh(() => fold([delegateStart("coder", "还在跑")]));
    expect(applyFilters(open, { ...DEFAULT_FILTERS, onlyProblems: true })).toHaveLength(1);
  });

  it("collectAgents 去重、保序，根与 specialist 都在", () => {
    expect(collectAgents(blocks)).toEqual(["supervisor", "coder"]);
  });
});

// ---------------------------------------------------------------- 摘要与边界

describe("summarizeArgs", () => {
  it("长字符串截断并报字符数（整份文件 content 必须截断，否则界面被撑爆）", () => {
    const content = "x".repeat(500);
    const text = summarizeArgs({ path: "m.py", content });
    expect(text).toContain("path=m.py");
    expect(text).toContain("500 字符");
    expect(text.length).toBeLessThan(300);
  });

  it("换行折成空格（参数摘要必须单行）", () => {
    expect(summarizeArgs({ content: "a\nb\nc" })).toBe("content=a b c");
  });

  it("非字符串值走 JSON；null 参数不抛", () => {
    expect(summarizeArgs({ replace_all: true, n: 3 })).toBe("replace_all=true, n=3");
    expect(summarizeArgs(null)).toBe("");
    expect(summarizeArgs({})).toBe("");
  });

  it("极多参数时总长也封顶", () => {
    const args = Object.fromEntries(Array.from({ length: 40 }, (_, i) => [`k${i}`, "v"]));
    expect(summarizeArgs(args).length).toBeLessThanOrEqual(121);
  });
});

describe("边界", () => {
  it("空事件流 → 空块列表", () => {
    expect(buildTimeline([])).toEqual([]);
  });

  it("未知事件类型不抛，退化成 step 行（后端加新 EventType 时界面不该白屏）", () => {
    const blocks = fresh(() =>
      fold([ev("未来事件" as EventType, "Supervisor", "???", { step: 1, node: "agent" })]),
    );
    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.kind).toBe("step");
  });

  it("只有图外事件时块照常渲染，且没有一条被判成重跑", () => {
    const blocks = fresh(() =>
      fold([
        ev("AgentStarted", "Supervisor", "", { step: null, node: null }),
        ev("AgentCompleted", "Supervisor", "", { step: null, node: null }),
      ]),
    );
    expect(blocks).toHaveLength(2);
    expect(blocks.every((b) => b.kind === "lifecycle" && !b.item.isRerun)).toBe(true);
  });

  it("同一个块 id 不重复（React key 靠它）", () => {
    const blocks = buildTimeline(markReruns(APPROVED));
    const walk = (list: Block[]): string[] =>
      list.flatMap((b) => [b.id, ...(b.kind === "delegation" ? walk(b.blocks) : [])]);
    const ids = walk(blocks);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("TimelineEvent 序列被原样保留（seq 用于排序与 React key）", () => {
    const items: TimelineEvent[] = markReruns(APPROVED);
    expect(items.map((i) => i.seq)).toEqual([...items].map((i) => i.seq).sort((a, b) => a - b));
  });
});
