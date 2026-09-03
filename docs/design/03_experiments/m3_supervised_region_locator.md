# M3：SLAKE-Supervised Region Locator

状态：`LOCATOR PILOT PASSED / ORACLE-CROP ANSWER CAUSAL SCREEN NEXT`

日期：2026-09-04

## 1. 已证伪问题与新假设

M2 的三条 zero-shot 路线依次失败：自主策略 16/16 在 overview 后停止；通用 forced decision 两次在首条走 null 出口；专用 locator 虽返回 region schema，却仍给出全图框。已确认的问题是 Qwen3.5-4B 缺少可靠的 question-to-region 输出能力，不是 crop executor 或 JSON 闭环损坏。

M3 的单一假设是：只对语言层做少量 QLoRA，用独立 detection 框监督工具 JSON/坐标 token，可在不改 backbone、视觉编码器、工具和 finalizer 的情况下建立可用定位策略。它不等价于答案提升；后者必须由独立 oracle/crop 因果臂验证。

## 2. 数据证据与隔离

来源固定为 `BoKelvin/SLAKE` revision `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`。官方图像包含 642 份 `detection.json`，580 份非空，共 1,622 个框、44 个标签。

builder 只读取 `q_lang`、`img_name`、`question`、图像尺寸/hash 和 `detection.json`；不读取 VQA `answer`。纳入规则：English；问题规范化文本恰好包含一个规范化 detection label；重复同名框取 union；归一化面积在 `[0.01,0.64]`。训练按图像最多 2 条、按标签最多 32 条；validation 按图像最多 1 条。

冻结 surface：

| Split | Rows | Unique images | Inference SHA-256 | Targets SHA-256 |
|---|---:|---:|---|---|
| train | 240 | 156 | `625099b7ef47a7605607847264e8555115cd8e223366d0b5f6ad2f0a4322f57a` | `41ac901413eb245568804e868e182543bc7d8495e9307dccd737edd3c5c78025` |
| validation | 43 | 43 | `96c1dd063882ce2f0694eeb92afccf019de0ec544ad7af84e3c5b28054abe18e` | `8bba7d746aad42fbe7a43c7fc289fa935657d726b22d8f3523ed355aa0a52fda` |

train/validation 的 image SHA 和 sample id 交集均为 0。inference 文件没有坐标/target/answer；targets 文件权限为 `0600`。`slake-train-locator-balanced-v1` 是 CLI 接线失败产生的未平衡废弃 surface，禁止训练；唯一允许训练的是 `slake-train-locator-balanced32-v1`。

## 3. 训练冻结项

- base：Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- quantization/compute：NF4 double quant + FP16；V100 SM70；eager attention
- trainable：仅 language-model attention/MLP LoRA；视觉编码器和 projector 冻结
- target：compact `region_inspect` JSON token；不训练答案、诊断或 final schema
- rank/alpha/dropout：`16/32/0.05`
- micro batch/gradient accumulation：`1/2`
- learning rate/scaler：`1e-4/1`
- image budget：`786432` pixels
- seed：`20260904`

## 4. 分阶段门禁

### M3-S0：2-step backward/save/reload smoke

命令：`scripts/run-slake-locator-qlora-smoke.sh`。必须满足 2/2 optimizer steps applied；loss 与 gradient norm 全有限；没有 scaler skip；保存 adapter 和 processor；adapter 文件哈希齐全；峰值显存小于 32GB。其后必须增加 adapter reload 的 4 条 inference smoke，不能只以 save 成功收口。

### M3-S1：冻结 validation 定位对照

在相同 43 条上运行 base locator 与 trained locator，推理时不读 targets；独立 scorer 最后读取 targets。报告 valid tool JSON、targeted area rate、mean IoU、IoU@0.3、IoU@0.5、label-stratified 结果、token、延迟和峰值显存。

首个 64-step pilot 只有同时满足以下条件才晋级：43/43 完成；valid/targeted rate ≥95%；相对 base 的 mean IoU 和 IoU@0.3 均提高；IoU@0.3 ≥40%。未通过不得搜索 seed；只允许从冻结错误类型中判定一次 objective/format 修复。

首次 64-step 输出已触发上述一次 format repair 权限：43 条中 24 条被判 invalid；固定首条最小重现显示模型实际给出合法坐标 `[577,577,800,700]`，但把本应恒定的工具名生成为 `inspect Pneumonia`。因此失败可归因于冗余常量字段，而非该条缺失坐标。

冻结修复 `M3-F1`：localizer 只生成 `{content, arguments}`，专用 controller 确定性封装 `name=region_inspect`。它不修复、裁剪或替换模型坐标，也不读取 target；图像、问题、训练选择、64 steps、seed 和超参数全部保持不变。由于 prompt/target contract 已变化，旧 base 和旧 adapter 结果只保留为诊断，F1 必须用新 commit 新建 base 与训练 run。F1 若仍未同时通过 valid/targeted、mean IoU 和 IoU@0.3 门，M3 停止，不允许第二次格式修复或增加训练步数。

F1 已通过。64-step 训练 run `qwen35-4b-slake-locator-f1-qlora-pilot64-s20260904` 完成 64/64 optimizer steps、128 examples，loss 全有限，mean/last `0.4940/0.4315`，峰值 CUDA `9,991.86 MiB`，adapter SHA-256 `ee865dd6871fc7da796af3c6ecc7abfe73b2e85fc33e817e0d8c3ee976e947fb`。

冻结 43 条 validation 对照：

| Arm | Valid | Targeted | Mean IoU | IoU@0.3 | IoU@0.5 | Prediction SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| F1 base | 100% | 72.09% | 0.1316 | 16.28% | 2.33% | `03c3e55220d63aea9d62b8933ee1316400a6d10087abedf066dcba11d282269e` |
| F1 locator-64 | 100% | 97.67% | 0.3299 | 58.14% | 25.58% | `80ca593a350b7a1ad353852bcdf0be1e25b35e144f296bd24e4d5c6ad61926a0` |

paired mean IoU `+0.1983`，IoU@0.3 `+41.86` points；34 条提高、2 条相同、7 条下降。满足 43/43、valid/targeted、mean IoU 与 IoU@0.3 的全部预注册门。该结论只限 question-conditioned 定位，不是答案或 Med-CMR 增益。

### M3-S2：先证实 crop 有用，再谈 Agent 效果

定位门通过后，先在 validation 上做 compute-matched `full-image`、`oracle-crop`、`black-crop` 答案对照。只有 oracle crop 相对 full-image 有正向 paired delta，且优于 black crop，才把 learned locator 接入 Agent。否则 locator 即使 IoU 高也不构成 benchmark 突破，路线停止。

该门已执行并失败：三臂均为 43/43，full-image / oracle-crop / black-crop 的 token F1 分别为 `68.95% / 59.69% / 16.07%`。oracle-crop 相对 full-image 为 `-9.26` 点，95% CI `[-20.74, 2.25]`；相对 black-crop 为 `+43.62` 点，95% CI `[26.39, 59.69]`。因此局部图包含真实答案信息，但以局部图替换全图会损失更多全局上下文。M3 crop-only 路线按预注册规则停止；后续独立 M4 改测 full + crop 融合，不把该失败改写成通过。

### M3-S3：Agent 因果臂

保持同一 finalizer 与两次模型调用，对比 learned crop、oracle crop、repeat overview、black crop；同时报告证据结构和答案指标。开发集通过后需多 seed 复现，才允许冻结一次新的 Med-CMR milestone；不得用 Med-CMR 逐题正误调 locator。

## 5. 资源

既有 answer-only QLoRA 在单张 V100 峰值约 6.9GB；locator target 更长，保守预计 8–12GB。1× V100-SXM2-32GB 足够 smoke 和 64-step pilot；无需多卡。若 target 序列导致 OOM，先降低 image pixel budget 并视为新的预注册 arm，不启用 CPU fallback。
