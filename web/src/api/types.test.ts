/**
 * 契约的前端侧钉子。`types.ts` 是纯类型，编译期能查的东西不需要测试 ——
 * **这里查的是编译期查不到的那一半**：真实后端 JSON 落进这些类型之后，
 * 字段名、`type` 的取值域、`step`/`node` 的 null 语义、`detail` 的形状是否真的对得上。
 *
 * 数据源是 `tests/test_web_ui_contract.py` 从真实会话导出的夹具（后端侧另有
 * `test_event_fields_match_types_ts` 钉字段名）。**两边一起钉**的理由：TS 的
 * `as unknown as` 断言能把任何形状骗过去，而契约漂移的真正代价是界面上某块
 * 静默空白 —— 不是编译错误。
 */
import { describe, expect, it } from "vitest";

import { APPROVED, BATCH, FIXTURE_TASK, REJECTED } from "../model/__fixtures__";
import {
  detailAs,
  type AgentEvent,
  type CondenseDetail,
  type EventType,
  type SessionInfo,
  type SessionList,
  type SessionStatus,
} from "./types";

/** `events/events.py` 的 EventType 九个值 —— 顺序与枚举一致（与 types.ts 同源）。 */
const EVENT_TYPES: EventType[] = [
  "AgentStarted",
  "AgentStep",
  "ToolCallStarted",
  "ToolCallCompleted",
  "ToolCallFailed",
  "AgentCompleted",
  "AgentFailed",
  "Condense",
  "TokenUsage",
];

/** `AgentEvent` 的全部字段（多一个少一个都要红）。 */
const EVENT_FIELDS = [
  "type",
  "agent",
  "message",
  "detail",
  "timestamp",
  "thread_id",
  "step",
  "node",
];

/** `SessionInfo` 的全部字段。 */
const SESSION_FIELDS = ["thread_id", "status", "approval", "result", "error", "event_count"];

const ALL = { APPROVED, REJECTED, BATCH };

describe("夹具本身是可用的真数据", () => {
  it("三份夹具都非空（否则下面所有断言都成了空转）", () => {
    for (const [name, rows] of Object.entries(ALL)) {
      expect(rows.length, `${name} 是空的`).toBeGreaterThan(0);
    }
    expect([APPROVED.length, REJECTED.length, BATCH.length]).toEqual([13, 7, 29]);
  });

  it("夹具是 **WS 帧**：每行都有 `kind: \"event\"`（`_normalize` 补的，不是手写的）", () => {
    for (const [name, rows] of Object.entries(ALL)) {
      rows.forEach((row, i) => {
        expect(row.kind, `${name}[${i}] 缺 kind`).toBe("event");
      });
    }
  });

  it("seq 从 0 起严格递增（回填与实时共用同一套序号）", () => {
    for (const [name, rows] of Object.entries(ALL)) {
      expect(rows.map((r) => r.seq), name).toEqual(rows.map((_, i) => i));
    }
  });
});

describe("AgentEvent 的字段名与取值域", () => {
  it("每条事件的字段集合与 `types.ts` 逐字一致", () => {
    for (const [name, rows] of Object.entries(ALL)) {
      rows.forEach((row, i) => {
        expect(Object.keys(row.event).sort(), `${name}[${i}]`).toEqual([...EVENT_FIELDS].sort());
      });
    }
  });

  it("`type` 全部落在九个值的联合里（后端加第十种事件时这里先红）", () => {
    const seen = new Set<string>();
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        expect(EVENT_TYPES).toContain(row.event.type);
        seen.add(row.event.type);
      }
    }
    // 夹具是刻意挑的场景，覆盖不到全部九种 —— 断言「见过的那几种」而不是全量
    expect([...seen].sort()).toEqual([
      "AgentCompleted",
      "AgentStarted",
      "AgentStep",
      "ToolCallCompleted",
      "ToolCallStarted",
    ]);
  });

  it("`agent` 是字符串且非空，`message` 是字符串（**可以为空串**，不是 null）", () => {
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        expect(typeof row.event.agent).toBe("string");
        expect(row.event.agent).not.toBe("");
        expect(typeof row.event.message).toBe("string");
      }
    }
  });

  it("`timestamp` 是秒级 ISO8601（`events.py:85` 的 `timespec=\"seconds\"`）", () => {
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        expect(row.event.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\+|-)\d{2}:\d{2}$/);
      }
    }
  });

  it("`thread_id` 是字符串（夹具里归一化成 `\"<thread>\"`）", () => {
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        expect(row.event.thread_id).toBe("<thread>");
      }
    }
  });
});

describe("step / node 的 null 语义", () => {
  it("`step` 要么是正整数，要么是 null（**没有 0** —— superstep 从 1 起）", () => {
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        const { step } = row.event;
        if (step !== null) {
          expect(Number.isInteger(step)).toBe(true);
          expect(step).toBeGreaterThan(0);
        }
      }
    }
  });

  it("`step === null` 与 `node === null` **同时**出现（图外事件的签名）", () => {
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        expect(row.event.step === null).toBe(row.event.node === null);
      }
    }
  });

  it("只有会话层发的根 AgentStarted / AgentCompleted 是图外事件", () => {
    const external: [string, string][] = [];
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        if (row.event.step === null) external.push([row.event.type, row.event.agent]);
      }
    }
    // 每份夹具各一条 AgentStarted + 一条 AgentCompleted，agent 都是 Supervisor
    expect(external).toEqual([
      ["AgentStarted", "Supervisor"],
      ["AgentCompleted", "Supervisor"],
      ["AgentStarted", "Supervisor"],
      ["AgentCompleted", "Supervisor"],
      ["AgentStarted", "Supervisor"],
      ["AgentCompleted", "Supervisor"],
    ]);
  });

  it("图内事件的 `node` 落在 agent / tools / condense 三个节点名里", () => {
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        if (row.event.node !== null) {
          expect(["agent", "tools", "condense"]).toContain(row.event.node);
        }
      }
    }
  });

  it("**只有 ToolCallStarted 带 detail** —— 工具结果原文拿不到（后端限制，详见 types.ts）", () => {
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        if (row.event.type === "ToolCallStarted") {
          expect(row.event.detail).not.toBeNull();
        } else {
          expect(row.event.detail, `${row.event.type} 竟然带了 detail`).toBeNull();
        }
      }
    }
  });
});

describe("detail 的形状（前端据此重建产出文件）", () => {
  it("delegate 的 detail.args 是 `{specialist, task}` 两个字符串", () => {
    const delegate = BATCH.filter(
      (r) => r.event.type === "ToolCallStarted" && r.event.detail?.args,
    ).filter((r) => "specialist" in (r.event.detail!.args as Record<string, unknown>));
    expect(delegate).toHaveLength(4); // 一批两个委派 + 重跑各一遍

    for (const row of delegate) {
      const args = row.event.detail!.args as Record<string, unknown>;
      expect(Object.keys(args).sort()).toEqual(["specialist", "task"]);
      expect(typeof args.specialist).toBe("string");
      expect(typeof args.task).toBe("string");
      expect(args.task).not.toBe("");
    }
  });

  it("`write_file` 的 args 带 `path` + **完整 content**（产出文件面板的全部依据）", () => {
    const writes = BATCH.filter((r) => r.event.detail?.args).filter(
      (r) => (r.event.detail!.args as Record<string, unknown>).path !== undefined,
    );
    // 3 条 = a.py（tester 首跑）+ a.py（整段重放，逐字相同）+ b.py（coder）
    expect(writes).toHaveLength(3);

    for (const row of writes) {
      const args = row.event.detail!.args as Record<string, unknown>;
      expect(Object.keys(args).sort()).toEqual(["content", "path"]);
      expect(args.content).toBe(`# ${String(args.path)}\n`);
    }
    expect(writes.map((r) => (r.event.detail!.args as Record<string, unknown>).path)).toEqual([
      "a.py",
      "a.py",
      "b.py",
    ]);
  });

  it("delegate 的 `specialist` 是小写注册表键，根 agent 是 `Supervisor`（大小写不统一是真源事实）", () => {
    for (const rows of Object.values(ALL)) {
      for (const row of rows) {
        const args = row.event.detail?.args as Record<string, unknown> | undefined;
        if (args && typeof args.specialist === "string") {
          expect(args.specialist).toBe(args.specialist.toLowerCase());
        }
      }
    }
    expect(APPROVED[0]!.event.agent).toBe("Supervisor");
    expect(APPROVED.some((r) => r.event.agent === "coder")).toBe(true);
  });

  it("夹具里的任务文本与 `FIXTURE_TASK` 一致（面板/卡片断言都引用它）", () => {
    const task = (APPROVED[2]!.event.detail!.args as Record<string, unknown>).task;
    expect(task).toBe(FIXTURE_TASK);
  });
});

describe("detailAs", () => {
  const event = (detail: Record<string, unknown> | null): AgentEvent => ({
    type: "Condense",
    agent: "Supervisor",
    message: "",
    detail,
    timestamp: "2026-01-01T00:00:00+00:00",
    thread_id: "t",
    step: 1,
    node: "condense",
  });

  it("null detail → null（不是 undefined，调用方 `?? 兜底` 两边都能接）", () => {
    expect(detailAs<CondenseDetail>(event(null))).toBeNull();
  });

  it("有 detail → **原样返回同一个引用**（不做拷贝、不做形状校验）", () => {
    const detail = { before: 40, after: 20, removed: 20, summary: "…" };
    expect(detailAs<CondenseDetail>(event(detail))).toBe(detail);
  });

  it("字段缺失时**不**伪造默认值 —— 收窄成 `T` 是调用方的承诺，不是运行时的保证", () => {
    const out = detailAs<CondenseDetail>(event({}));
    expect(out).toEqual({});
    expect(out?.before).toBeUndefined();
  });
});

describe("SessionInfo 的字段名（与后端 `test_session_info_fields_match_types_ts` 对齐）", () => {
  /**
   * 标注成 `SessionInfo` 才是真的钉：接口**加一个必填字段**这里就编译不过，
   * 而 `SESSION_FIELDS` 又必须跟着改 —— 三个地方（`types.ts` / 这里 / 后端的
   * `SESSION_FIELDS`）因此只能一起动。不标注的话这就只是一段自说自话的字面量。
   */
  const SAMPLE: SessionInfo = {
    thread_id: "t",
    status: "idle",
    approval: [],
    result: null,
    error: null,
    event_count: 0,
  };

  it("字段集合与 SESSION_FIELDS 一致", () => {
    expect(Object.keys(SAMPLE).sort()).toEqual([...SESSION_FIELDS].sort());
  });

  it("五态字面量与后端 `SessionStatus.value` 对齐", () => {
    // 第五态 `interrupted` 是 P9 加的（`runtime/session.py`），后端侧由
    // `test_web_ui_contract.py` 对着枚举钉；这里钉的是 TS 这一半别漏。
    const all: SessionStatus[] = [
      "idle",
      "running",
      "awaiting_approval",
      "interrupted",
      "closed",
    ];
    expect(all).toHaveLength(5);
    for (const s of all) SAMPLE.status = s;
  });

  it("SessionList 带 history_available（P9 决定④的能力位）", () => {
    // 标注成 `SessionList` 才是真的钉：后端去掉这个键 → 这里编译不过。
    const list: SessionList = { sessions: [SAMPLE], history_available: true };
    expect(Object.keys(list).sort()).toEqual(["history_available", "sessions"]);
  });
});
