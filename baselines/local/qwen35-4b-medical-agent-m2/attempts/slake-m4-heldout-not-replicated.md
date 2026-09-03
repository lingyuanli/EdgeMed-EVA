# SLAKE M4 official-test held-out：NOT REPLICATED

日期：2026-09-04  
性质：一次性 external held-out；不是 Med-CMR official score

## 隔离与完整性

- SLAKE revision：`a9083ce6c34ac3ffb17671a605962924d8a8f9e9`
- test source SHA-256：`6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3`
- locator inference SHA-256：`a32e4c08ebab9bf83b890e5db41b5d3fbd0f07fd58bc5f418309cf0d1ee77bef`
- 45 条、45 张图；与 locator train 和 validation 的 image SHA/sample id 均零重叠。
- locator predictions 在读取 targets/答案前完成并封存：`53b156bb58abd6e944a1c674e5a00e6dea0bd230b686d9f103df546335decb77`。
- answer references SHA-256：`30da688ca8c5644b27d45acb1dcbf21b500aad66c536124de577088f7822ecfd`，mode 0600。
- 四个答案臂均 45/45 completed，无失败样本。

## Locator held-out

- valid output：100%；targeted：100%。
- mean IoU：`0.3751`；IoU@0.3：`68.89%`；IoU@0.5：`33.33%`。
- inference time：`312.65 s`；peak allocated：`3,586.68 MiB`。

## Answer held-out

| Arm | Exact | Token F1 | Predictions SHA-256 |
|---|---:|---:|---|
| full | 55.56% | 68.54% | `e86df9382f2336de6ffbbbc30068579156fad08ee5d6cd01b34bc357a1823f06` |
| full + black | 57.78% | 69.72% | `94519b551bf43f97265876d065960a3fddb57a86e439e78b83c1afac9d9bf7dc` |
| full + learned | 57.78% | 69.84% | `028be341949d5c5ce0d36d2e6279e394d350c302ed0506b365b5b378b9905d3a` |
| full + oracle | 55.56% | 66.54% | `675470012e432d0486b1e9a8da77c25f52e7538d2f83997af300529adfd7eece` |

- learned - full：token F1 `+1.31`，95% CI `[-6.26, 9.37]`；exact `+2.22`。
- learned - black：token F1 `+0.12`，95% CI `[-0.09, 0.44]`；exact `0.00`。
- oracle - full：token F1 `-2.00`，95% CI `[-11.35, 7.10]`；exact `0.00`。

## 判定

定位器跨 split 泛化，但多视图答案机制没有复现：learned 与黑图几乎等价，oracle 方向反转。validation 上的 `+1.98` 只能保留为未复现 pilot，不构成可部署增益。按冻结规则关闭 crop-fusion，不再使用 SLAKE test 进行任何选择、prompt 或超参调整，也不进入 Med-CMR milestone。
