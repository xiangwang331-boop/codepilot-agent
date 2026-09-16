/**
 * 最终结果的分段渲染。
 *
 * `AgentState.result` 是模型给的最终回答，通常混着散文与 ``` 围栏代码块。
 * 这里把它切成段落，交给 React 当**文本节点**渲染 —— 全程不用
 * `dangerouslySetInnerHTML`。模型输出是不可信内容（它读过 workspace 里的文件、
 * 跑过任意命令），拼 HTML 就是自找 XSS。
 */

export type ResultSegment =
  | { kind: "text"; text: string }
  | { kind: "code"; lang: string; code: string };

const FENCE = /^```(.*)$/;

/**
 * 按 ``` 围栏切分。
 *
 * 容错：**未闭合的围栏**把剩余全部当代码，照常返回（绝不抛）——
 * 会话被中断时结果里留一个半截围栏是很常见的。
 *
 * **空段落一律丢弃**：空串输入、或者围栏紧贴开头/结尾，都会产出空 buffer。
 * 留着它们渲染成 `<pre></pre>`，在界面上就是一条凭空的空隙（`pre` 有 margin）——
 * 而 `ResultPanel` 是按段渲染的，没有别的地方能过滤掉。
 */
export function splitFencedCode(text: string): ResultSegment[] {
  const lines = text.split("\n");
  const out: ResultSegment[] = [];
  let buffer: string[] = [];
  let inCode = false;
  let lang = "";

  const flushText = (): void => {
    const joined = buffer.join("\n");
    buffer = [];
    if (joined !== "") out.push({ kind: "text", text: joined });
  };

  const flushCode = (): void => {
    const code = buffer.join("\n");
    buffer = [];
    if (code !== "") out.push({ kind: "code", lang, code });
  };

  for (const line of lines) {
    const m = FENCE.exec(line.trim());
    if (m) {
      if (inCode) {
        flushCode();
        inCode = false;
        lang = "";
      } else {
        flushText();
        inCode = true;
        // 4 个及以上反引号也是合法围栏（````md）—— 只取前三个当定界符，
        // 剩下的反引号不能漏进 lang，否则代码块头上会显示 "`md"。
        lang = (m[1] ?? "").trim().replace(/^`+/, "");
      }
      continue;
    }
    buffer.push(line);
  }

  if (inCode) {
    // 未闭合：剩余内容当代码，不丢
    flushCode();
  } else {
    flushText();
  }
  return out;
}
