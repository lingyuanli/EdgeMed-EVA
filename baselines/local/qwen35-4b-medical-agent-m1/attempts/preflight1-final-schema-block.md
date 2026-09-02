# M1 real preflight-1：final schema 阻断

日期：2026-09-02

状态：`INFERENCE_COMPLETED / VERIFIER_BLOCK`

用途：真实模型 operational wiring；不是 efficacy evaluation。

## 冻结条件

- code commit：`2eff8654cebe9ec557274b7783aaf3c671b57535`
- model：`Qwen/Qwen3.5-4B`
- model revision：`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- model source manifest SHA-256：`a8bfc09b80581bd5d74065ca9574da513433e46b6ed117bad19c5858f2d03def`
- 两片权重：字节数和 SHA-256 均通过 source manifest 校验
- hardware：`Tesla V100-SXM2-32GB`
- inference surface：PMC-VQA 冻结 inference surface 的首条记录；runner 不读取 reference
- generation：NF4 double quant、FP16、eager、greedy、thinking disabled
- tool allowlist：`inspect_overview`、`region_inspect`
- max steps：2；decision/final token 上限：192/512
- run：`runs/qwen35-4b-medical-agent-m1-pmc-preflight1-20260902`
- run contract SHA-256：`ae715cdf8f74d0a85643817cc3cf289bc6e6152e4065ff533f4d362e21ac62d1`

## 运行事实

- 1/1 inference completed；模型加载和两轮 decision 均完成。
- 模型自主调用一次 `region_inspect`，工具状态 `completed`，输入图像和输出 crop 均有哈希。
- finalizer 生成了严格可解析 JSON，且 MCQ 选项为 `C`；该单题正确性不用于调参或效果主张。
- final evidence 使用 `id` 代替 `evidence_id`；`supports/contradicts` 为字符串而不是数组；`acquisition` 不是工具枚举；hypothesis 缺少 `label/status`。

## 独立评测

- reference 只在 inference 完成后由 `finalize_agent_eval` 接入。
- E0：schema valid `0.0`，citation valid `0.0`，tool-trace bound `1.0`。
- E1：1/1 仅为 operational observation，不构成 efficacy evidence。
- E3 与 medical correctness：`DEFER`。
- 六项完整性/复算检查 PASS；声明式质量门 BLOCK；总 verdict `BLOCK`，finalize 命令退出码 1。

## 单变量修订

只修改 final schema contract：提供完整字段名、类型、枚举和 JSON skeleton，同时让 scorer 严格验证 hypothesis/evidence 结构及 acquisition-trace 对齐。工具、样本、模型、量化、解码预算和答案策略保持不变。旧 run 不覆盖，以新 commit 和新 run id 重试。
