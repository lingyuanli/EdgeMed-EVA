# M2 P3 dedicated localizer gate

日期：2026-09-04

状态：`INFERENCE COMPLETED / TARGETED ROI GATE BLOCK / NO REFERENCE SCORING`

- code commit：`bd73ef9b216d9c544f5af26c11b3c2e9ddd6624f`
- run：`runs/qwen35-4b-medical-agent-m2-slake1-dedicated-localizer-20260904`
- run contract SHA-256：`b7b379e7d03c24997303aee402bac42f192af956f5ea2d27a674918d7f7a5c33`
- prompt contract SHA-256：`9d54f998acd857c660c378de0f2d127aca05a5b646668673d7a1db24525869d4`
- 1/1 inference complete；overview/region 各 1 次，工具失败 0；E0 schema/citation/trace 全 1
- localizer 原始输出 region：`[0,0,1000,1000]`；面积 `1.0`
- targeted ROI：`0/1`；reference-free operational analyzer：`BLOCK`
- 2 次模型调用；input/output tokens `3,137/366`；模型调用延迟 `28.9194s`
- peak CUDA allocated/reserved：`4,475.76/4,826 MiB`

专用 schema 成功消除了 P2 的 null 退出，但没有产生空间定位：模型把 full-frame 当作 region 返回，并给出占位 target `concrete visual distinction`。因此 16 条扩展按预注册停止，不能将本结果解释为 crop 有效或答案正确。

下一路线不再修改 zero-shot prompt。SLAKE 官方图像包中的 `detection.json` 提供独立框监督；M3 先验证 locator QLoRA 的梯度/保存闭环，再在完全隔离的 validation 图像上评测定位，之后才允许做 oracle crop 的答案因果筛选。
