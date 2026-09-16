import { Panel } from "./Panel";
import { agentLabel } from "../model/agent";
import { formatTokens, type TokenSummary } from "../model/tokens";

/**
 * 消耗统计 —— `main.py` 收尾那段打印的图形版。
 *
 * ⚠️ 数据源 `TokenUsage` 只有**真实 LLM** 才发（`core.py:64` 读
 * `response_metadata["token_usage"]`）。用 FakeLLM 跑的会话这里是空的，不是 bug。
 */
export function TokenPanel({ summary }: { summary: TokenSummary }): React.JSX.Element {
  const { total, byAgent } = summary;

  return (
    <Panel title="消耗统计" count={total.calls > 0 ? `${total.calls} 次调用` : undefined} defaultOpen={total.calls > 0}>
      {total.calls === 0 ? (
        <div className="panel__empty" style={{ padding: 0 }}>
          本会话还没有 LLM 消耗记录。
          <br />
          （只有真实模型调用才会上报；用假 LLM 驱动时不会产生。）
        </div>
      ) : (
        <table className="tokentable">
          <thead>
            <tr>
              <th>agent</th>
              <th>调用</th>
              <th>输入</th>
              <th>输出</th>
              <th>合计</th>
            </tr>
          </thead>
          <tbody>
            {byAgent.map((b) => (
              <tr key={b.agent}>
                <td>{agentLabel(b.agent)}</td>
                <td>{b.calls}</td>
                <td>{formatTokens(b.prompt)}</td>
                <td>{formatTokens(b.completion)}</td>
                <td>{formatTokens(b.total)}</td>
              </tr>
            ))}
            <tr data-total="true">
              <td>总计</td>
              <td>{total.calls}</td>
              <td>{formatTokens(total.prompt)}</td>
              <td>{formatTokens(total.completion)}</td>
              <td>{formatTokens(total.total)}</td>
            </tr>
          </tbody>
        </table>
      )}
    </Panel>
  );
}
