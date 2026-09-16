import { agentLabel } from "../model/agent";
import type { FilterState } from "../model/events";

/**
 * 时间线过滤器。刻意只给三个开关 —— 事件类型矩阵那种 UI 比数据本身还复杂。
 *
 * `TokenUsage` 默认**关**：一次真实会话 27 次 LLM 调用就是 27 行噪音，
 * 聚合数字在右侧的「消耗统计」面板里看。
 */
export function FilterBar({
  filters,
  agents,
  onChange,
}: {
  filters: FilterState;
  agents: string[];
  onChange: (next: FilterState) => void;
}): React.JSX.Element {
  return (
    <div className="filters">
      <select
        value={filters.agent}
        onChange={(e) => onChange({ ...filters, agent: e.target.value })}
        title="只看某个 agent 的动作"
      >
        <option value="all">全部 agent</option>
        {agents.map((a) => (
          <option key={a} value={a}>
            {agentLabel(a)}
          </option>
        ))}
      </select>

      <label title="一次真实会话会有几十条 LLM 调用记录，默认折叠">
        <input
          type="checkbox"
          checked={filters.showTokenUsage}
          onChange={(e) => onChange({ ...filters, showTokenUsage: e.target.checked })}
        />
        显示 token 明细
      </label>

      <label title="只保留失败的事件、失败的委派和未收尾的工具调用">
        <input
          type="checkbox"
          checked={filters.onlyProblems}
          onChange={(e) => onChange({ ...filters, onlyProblems: e.target.checked })}
        />
        只看问题
      </label>
    </div>
  );
}
