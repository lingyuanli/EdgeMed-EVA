# SLAKE M4 learned multiview validation pilot

日期：2026-09-04  
判定：`pilot_signal_only`；不是 Med-CMR official score

## 因果绑定

- locator predictions：`80ca593a350b7a1ad353852bcdf0be1e25b35e144f296bd24e4d5c6ad61926a0`
- locator run：`qwen35-4b-slake-locator-f1-pilot64-43-20260904`
- learned multiview manifest：`f5574534d50c02b911158425c4e9ed472b8dad48c03d6ca859787ef01da7a52d`
- answer references：`f47b05c2786b3c93e4b61e52102119a84c2d4a2c603e4b5fe582f1cb5e316f0e`
- answer predictions：`cdf4f6d3763de708c5a6bbac02a6ce9d4bef0713caaf97ad0ea636e8870cd140`
- inference contract：`c14d37913b14a1a4aabb37b8be3f465aac1d6bfa2611e6487e23917ca8f1f65a`
- code：`e005637`

构建器读取 completed locator run manifest 与 predictions，不读取 locator targets、答案或 references。回答阶段使用 base Qwen3.5-4B，不加载 locator adapter。

## 结果

- 43/43 completed；峰值 CUDA `4,480.40 MiB`；推理 `42.17 s`。
- learned multi-view：exact `53.49%`，token F1 `70.93%`。
- learned - full：token F1 `+1.98`，95% CI `[0.00, 5.08]`；exact `0.00`。
- learned - full+black：token F1 `+2.76`，95% CI `[-4.65, 11.63]`；exact `+2.33`。
- learned - oracle：token F1 `-5.62`，95% CI `[-14.49, 1.36]`。
- oracle 增量捕获：`26.06%`，超过预注册 25% 点估计门。

## 边界与下一步

点估计支持 learned locator 为多视图回答带来小幅增益，但 compute-matched black 对照区间跨零。不得据此声明稳健提升或访问 Med-CMR 调参。下一步仅执行一次已冻结的 SLAKE official test held-out replication；失败后不基于 test 调参。
