/**
 * Agent 名字的归一化与显示。
 *
 * ⚠️ 大小写不统一是**真源的事实**，不是笔误：根 agent 硬编码为 `"Supervisor"`
 * （`runtime/session.py`），specialist 是小写注册表键。所以比大小写敏感就必然出错
 * —— `[data-agent="coder"]` 的 CSS、过滤器的比较、`DelegationBlock.specialist`
 * 的对齐，全都依赖这一层归一化。
 */
import { describe, expect, it } from "vitest";

import { agentLabel, agentRole, agentSlug } from "./agent";

describe("agentSlug", () => {
  it("根 agent 的大写被吃掉", () => {
    expect(agentSlug("Supervisor")).toBe("supervisor");
  });

  it("注册表里的 6 个 specialist 原样通过", () => {
    for (const name of ["analyst", "planner", "coder", "debugger", "tester", "reviewer"]) {
      expect(agentSlug(name)).toBe(name);
    }
  });

  it("首尾空白被去掉", () => {
    expect(agentSlug("  Coder\n")).toBe("coder");
  });

  it("非字母数字折成连字符 —— 结果必须是**合法 CSS 属性值**", () => {
    // 注册表将来加一个 "code-reviewer" 之类的角色不该让 `[data-agent=…]` 失效
    expect(agentSlug("Code Reviewer")).toBe("code-reviewer");
    expect(agentSlug("a.b/c")).toBe("a-b-c");
  });

  it("首尾连字符被剥掉（`-coder-` 不该出现）", () => {
    expect(agentSlug("__coder__")).toBe("coder");
  });

  it("**非 ASCII 一律折掉** —— 所以中文占位符 slug 成空串（走 fallback 灰）", () => {
    // `events.ts` 在 delegate 的 args 缺 specialist 时填的是 `"(未知)"`。
    // 它**不进 LABELS**，但 `agentLabel` 会把原串透传出去，所以界面上显示的仍是
    // 「(未知)」而不是空白 —— 归一是为了 CSS 属性安全，不是为了显示。
    expect(agentSlug("(未知)")).toBe("");
    expect(agentLabel("(未知)")).toBe("(未知)");
  });

  it("空串 / 全是符号 → 空串（调用方据此走 fallback 灰色）", () => {
    expect(agentSlug("")).toBe("");
    expect(agentSlug("!!!")).toBe("");
  });

  it("幂等：归一化两次与一次相同（过滤链路里会被反复调用）", () => {
    for (const s of ["Supervisor", "Code Reviewer", "  coder  ", "!!!", ""]) {
      expect(agentSlug(agentSlug(s))).toBe(agentSlug(s));
    }
  });
});

describe("agentLabel", () => {
  it("已知角色给规范大小写（界面上一律 'Coder'，不是 'coder'）", () => {
    expect(agentLabel("coder")).toBe("Coder");
    expect(agentLabel("Supervisor")).toBe("Supervisor");
    expect(agentLabel("CODER")).toBe("Coder");
  });

  it("未注册的 agent **原样返回** —— 注册表加新角色不该让界面崩或显示 undefined", () => {
    expect(agentLabel("architect")).toBe("architect");
  });

  it("空串原样返回（不显示 'undefined'）", () => {
    expect(agentLabel("")).toBe("");
  });
});

describe("agentRole", () => {
  it("已知角色给中文职责（徽章 title）", () => {
    expect(agentRole("coder")).toContain("写代码");
    expect(agentRole("Supervisor")).toContain("编排");
  });

  it("未注册的给一句兜底文案，不是空串", () => {
    expect(agentRole("architect")).toBe("未注册的 agent");
  });
});
