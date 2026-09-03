# M3 F1 supervised locator pilot

日期：2026-09-04

状态：`LOCALIZATION GATE PASS / ANSWER EFFICACY UNMEASURED`

- code commit：`b4fef2df4add61350df94d98490db44bb3fdaa7f`
- train surface：240 rows / 156 train images；inference SHA `625099b7…`，targets SHA `41ac9014…`
- validation surface：43 rows / 43 images；与 train image SHA overlap 为 0
- training run：`runs/qwen35-4b-slake-locator-f1-qlora-pilot64-s20260904`
- 64/64 optimizer steps applied；128 examples；finite mean/last loss `0.4939675/0.4314511`
- peak CUDA：`9,991.86 MiB`
- adapter SHA-256：`ee865dd6871fc7da796af3c6ecc7abfe73b2e85fc33e817e0d8c3ee976e947fb`
- base run：`runs/qwen35-4b-slake-locator-f1-base43-20260904`
- trained run：`runs/qwen35-4b-slake-locator-f1-pilot64-43-20260904`

| Metric | Base | Locator-64 | Delta |
|---|---:|---:|---:|
| valid output | 100% | 100% | 0 |
| targeted area | 72.09% | 97.67% | +25.58 pt |
| mean IoU | 0.1316 | 0.3299 | +0.1983 |
| IoU@0.3 | 16.28% | 58.14% | +41.86 pt |
| IoU@0.5 | 2.33% | 25.58% | +23.26 pt |

逐样本：34 improved / 2 tied / 7 worsened。该结果通过 M3-S1 定位门，但不能证明 crop 改善医学答案。下一步必须先跑 full/oracle-crop/black-crop 的冻结答案因果对照；未通过则不能把 locator 接入 Agent 效果主张。
