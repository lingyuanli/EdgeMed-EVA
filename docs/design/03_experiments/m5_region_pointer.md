# M5：保留全图的区域指示器

## 1. 动机与边界

M4 证明 locator 能在未见 test 图像上保持定位指标，但 crop-fusion 答案增益未复现。M5 不使用 SLAKE test 的逐题或聚合错误做选择，只从已知的结构性限制出发：crop 会改变视野和多图输入分布。新机制在原尺寸完整图像上仅叠加一个可见矩形，不裁剪、不增加图像数量。

## 2. M5-S1 oracle pointer 因果门

- 开发数据：已冻结 SLAKE validation 43 条；不再访问 official test。
- 模型/回答：base Qwen3.5-4B、原 `answer_only` prompt、greedy、32 tokens、786,432 pixels。
- Arm A：既有 full-image 基线。
- Arm B：完整图 + oracle 红色矩形。
- Arm C：完整图 + sham 红色矩形；将 43 个 oracle boxes 按 sample-id 排序循环移位一位，保持框坐标/面积分布，但打破问题—位置对应。
- 绘制：RGB `(255,0,0)`；线宽 `max(2, round(min(width,height)*0.008))`；不修改问题文本。
- 主比较：B-C 的 token F1 paired delta，10,000 次 bootstrap，seed `20260904`。
- 次比较：B-A；报告 exact、token F1、完整性、图像与 prediction SHA-256。

只有 B-C 点估计大于 0 且 95% CI 下界大于 0，才允许构建 learned-pointer。若 B-A 不大于 0，则即使 B-C 为正也只视为机制信号，必须另行设计无提示损伤的呈现方式。任一门失败时停止 pointer，不根据 validation 单题移动或改画框。
