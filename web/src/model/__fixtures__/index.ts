/**
 * 前端测试夹具：**后端真正发出来的事件流**。
 *
 * 由 `tests/test_web_ui_contract.py` 从真实会话（假 LLM，零 token 成本）导出并归一化
 * （`thread_id` → `"<thread>"`，`timestamp` → 按 seq 递增的固定串）。**不要手改** ——
 * 改了就跟真实发射顺序脱节，那正是这些夹具存在的唯一理由。
 *
 * 两个夹具各覆盖折叠算法的一条主分支：
 *
 * | 文件 | 场景 | 覆盖的分支 |
 * |---|---|---|
 * | `approval-approved.json` | 批准 → coder 真跑了、写了文件 | `completed` + `attempts === 2` |
 * | `approval-rejected.json` | 拒绝 → 子 agent 从未启动 | `not_executed` + `childRuns === 0` |
 * | `batch-two-delegates.json` | 一批两个委派、只批第二个 | 悬空开块 + 子事件整段重放 |
 *
 * 两者的前 4 条事件**逐字相同**（seq 0/1 生命周期与 step、seq 2/3 挂起重跑的
 * `ToolCallStarted(delegate)`）—— 这正是「挂起」在同一份数据里表现为「同一块
 * `attempts++`」而不是「新开一块」的实证。
 */
import type { EventEnvelope } from "../../api/types";

import approvedRaw from "./approval-approved.json";
import batchRaw from "./batch-two-delegates.json";
import rejectedRaw from "./approval-rejected.json";

/**
 * JSON 导入的推断类型是宽化的（`type` 是 `string` 而不是九个字面量的联合），
 * 所以必须过一次 `unknown` 才能收窄到契约类型。
 *
 * 行形状是 **WS 帧**（`{kind:"event", seq, event}`）而不是 REST 回填行
 * （`{seq, event}`）—— 前端在真实路径上拿到的就是前者，所以夹具补了 `kind`。
 * 这条由 `api/types.test.ts` 钉住。
 */
function envelopes(raw: { events: unknown }): EventEnvelope[] {
  return raw.events as unknown as EventEnvelope[];
}

/** 批准路径：委派真的执行了，coder 写了一个文件。 */
export const APPROVED = envelopes(approvedRaw);

/** 拒绝路径：委派以**零个子事件**闭合。 */
export const REJECTED = envelopes(rejectedRaw);

/**
 * 一批两个委派（tester 写 a.py + coder 写 b.py），只有第二个需要批准。
 *
 * 这是折叠算法最难的形状，两条都在这里：**第二个委派的开块事件收不了口**
 * （`interrupt()` 在它的执行体里抛出，tools 节点的循环就地中断），**第一个委派连同
 * 子事件整段重放** —— 顺序是 `S(a) C(a) S(b) | S(a) C(a) S(b) C(b)`。
 */
export const BATCH = envelopes(batchRaw);

/** 夹具里的任务文本（两条流程相同）。 */
export const FIXTURE_TASK = "写一个 quicksort 到 main.py";
