/**
 * 最终结果的分段。**这套切分是「不用 `dangerouslySetInnerHTML`」的前提** ——
 * 模型输出是不可信内容（它读过 workspace 里的文件、跑过任意命令），
 * 切成段落交给 React 当文本节点渲染，HTML 就永远是字面量。
 */
import { describe, expect, it } from "vitest";

import { splitFencedCode } from "./result";

describe("splitFencedCode", () => {
  it("纯散文 → 一段 text", () => {
    expect(splitFencedCode("做好了。\n没有代码。")).toEqual([
      { kind: "text", text: "做好了。\n没有代码。" },
    ]);
  });

  it("空串 → 零段（不是一段空 text）", () => {
    expect(splitFencedCode("")).toEqual([]);
  });

  it("带语言的围栏 → text / code / text 三段，lang 提取出来", () => {
    const out = splitFencedCode("看：\n```python\nx = 1\n```\n好了。");
    expect(out).toEqual([
      { kind: "text", text: "看：" },
      { kind: "code", lang: "python", code: "x = 1" },
      { kind: "text", text: "好了。" },
    ]);
  });

  it("无语言围栏 → lang 是空串", () => {
    const out = splitFencedCode("```\nraw\n```");
    expect(out).toEqual([{ kind: "code", lang: "", code: "raw" }]);
  });

  it("代码块在开头 → 没有前导空 text 段", () => {
    const out = splitFencedCode("```py\nx\n```\n尾注");
    expect(out.map((s) => s.kind)).toEqual(["code", "text"]);
  });

  it("多个代码块（多文件产出是常态）", () => {
    const out = splitFencedCode("```py\na\n```\n中间\n```js\nb\n```");
    expect(out.map((s) => s.kind)).toEqual(["code", "text", "code"]);
    expect(out.filter((s) => s.kind === "code")).toHaveLength(2);
  });

  it("**未闭合的围栏**把剩余全部当代码 —— 会话被中断时结果里留半截围栏很常见", () => {
    const out = splitFencedCode("开始\n```python\n还没写完");
    expect(out).toEqual([
      { kind: "text", text: "开始" },
      { kind: "code", lang: "python", code: "还没写完" },
    ]);
  });

  it("围栏行两端空白被容忍（模型经常多打空格）", () => {
    const out = splitFencedCode("  ```py  \nx\n  ```  ");
    expect(out).toEqual([{ kind: "code", lang: "py", code: "x" }]);
  });

  it("代码块内的 ``` 会闭合（不支持嵌套，与 Markdown 一致）", () => {
    const out = splitFencedCode("```\na\n```\nb\n```\nc\n```");
    expect(out.map((s) => s.kind)).toEqual(["code", "text", "code"]);
  });

  it("代码块内容里的反引号不会被当成围栏（只有整行以 ``` 开头才算）", () => {
    const out = splitFencedCode("```\nlet s = `a${b}`\n```");
    expect(out).toEqual([{ kind: "code", lang: "", code: "let s = `a${b}`" }]);
  });

  it("HTML / script 当**字面量**留在 text 里（不做任何转义或剥离，交给 React）", () => {
    const evil = "<img src=x onerror=alert(1)>";
    const out = splitFencedCode(evil);
    expect(out).toEqual([{ kind: "text", text: evil }]);
  });

  it("四反引号（````）也按围栏处理，且 lang 不带多余的反引号", () => {
    const out = splitFencedCode("````md\n# 标题\n````");
    expect(out).toEqual([{ kind: "code", lang: "md", code: "# 标题" }]);
  });

  it("只有空行 → 一段 text 装着空行（不丢）", () => {
    expect(splitFencedCode("\n\n")).toEqual([{ kind: "text", text: "\n\n" }]);
  });

  it("**空段落被丢掉**，不留 `pre` 空壳（不然界面上多一道凭空的空隙）", () => {
    // 围栏紧贴开头：`flushText` 面对的是空 buffer，不该产出空 text 段
    expect(splitFencedCode("```py\nx\n```").map((s) => s.kind)).toEqual(["code"]);
    // 围栏紧贴结尾：不该产出尾部空 text 段
    expect(splitFencedCode("x\n```py\n```").map((s) => s.kind)).toEqual(["text"]);
    // 空代码块整个消失，而不是留一个空的 codeblock 外壳
    expect(splitFencedCode("a\n```py\n```\nb").map((s) => s.kind)).toEqual(["text", "text"]);
  });

  it("未闭合围栏且后面什么都没有 → 只剩前面的 text", () => {
    expect(splitFencedCode("只说一半\n```py\n")).toEqual([{ kind: "text", text: "只说一半" }]);
  });
});
