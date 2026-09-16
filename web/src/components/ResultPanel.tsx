import { Panel } from "./Panel";
import { splitFencedCode } from "../model/result";

/**
 * 最终结果。
 *
 * **全程不用 `dangerouslySetInnerHTML`**：模型输出是不可信内容（它读过 workspace
 * 里的文件、跑过任意命令），拼 HTML 就是自己给自己开 XSS。代码块走 React 文本节点。
 *
 * `resultText` 为 null 不代表出错 —— `awaiting_approval` 的挂起路径上
 * `session.py:445-452` 会在置 result 之前就 return，所以挂起时这里本来就是空的。
 */
export function ResultPanel({ text }: { text: string | null }): React.JSX.Element {
  return (
    <Panel title="最终结果">
      {text === null ? (
        <div className="panel__empty" style={{ padding: 0 }}>
          暂无 —— 任务还在跑、或正等着批准。
        </div>
      ) : (
        splitFencedCode(text).map((seg, i) =>
          seg.kind === "text" ? (
            <pre className="result__text" key={i}>
              {seg.text}
            </pre>
          ) : (
            <div className="codeblock" key={i}>
              {seg.lang !== "" && <div className="codeblock__lang">{seg.lang}</div>}
              <pre>{seg.code}</pre>
            </div>
          ),
        )
      )}
    </Panel>
  );
}
