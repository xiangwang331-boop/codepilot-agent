/**
 * Agent 名字归一化与显示名。
 *
 * ⚠️ 大小写不统一是真实情况，不要「顺手统一」：
 *   - 根 agent 是 **`"Supervisor"`**（首字母大写，硬编码在 `runtime/session.py:427/440/460`
 *     与 `runtime/assembly.py`）
 *   - specialist 是小写 `spec.name`（`agent/specialists.py` 的注册表键：
 *     `analyst` / `planner` / `coder` / `debugger` / `tester` / `reviewer`）
 *
 * 所以 CSS 的 `[data-agent="coder"]` 要用 slug，过滤器的比较也要小写化。
 */

/** 归一化成 CSS 属性安全、比较安全的 slug。`"Supervisor"` → `"supervisor"`。 */
export function agentSlug(agent: string): string {
  return agent
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

const LABELS: Record<string, string> = {
  supervisor: "Supervisor",
  analyst: "Analyst",
  planner: "Planner",
  coder: "Coder",
  debugger: "Debugger",
  tester: "Tester",
  reviewer: "Reviewer",
};

/** 显示名。未注册的 agent 原样返回（注册表加新角色不该让界面崩）。 */
export function agentLabel(agent: string): string {
  const slug = agentSlug(agent);
  return LABELS[slug] ?? agent;
}

/** 每个 agent 的中文职责，用于徽章的 title 提示。 */
const ROLES: Record<string, string> = {
  supervisor: "编排者：拆解需求并委派给 specialist",
  analyst: "需求分析（只读）",
  planner: "实现规划（只读）",
  coder: "写代码（需批准）",
  debugger: "定位问题（只读 + 诊断命令）",
  tester: "跑测试验证",
  reviewer: "代码审查",
};

export function agentRole(agent: string): string {
  return ROLES[agentSlug(agent)] ?? "未注册的 agent";
}
