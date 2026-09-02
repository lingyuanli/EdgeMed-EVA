# A1 Semantic Option-Order Consensus Agent

Status: `completed / operational checks passed / accuracy gate failed / archived`

Date: 2026-09-02

Frozen parent: M2a seed-20260903 adapter

## Result

The two missing cyclic views each completed 512/512 and the full campaign exited zero. The controller mapped all four visible option orders back to canonical semantic option identities and produced 512 auditable Agent predictions.

| Comparison | A | A1 | A1 - A | Paired 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| frozen B0 vs A1 | 58.0078% | 58.0078% | 0.0000 | [-4.6875, 4.6875] | 1.0000 |
| M2a single-view vs A1 | 62.8906% | 58.0078% | -4.8828 | [-8.3984, -1.5625] | 0.00728 |

A1 returned 297/512 correct and four invalid outputs. It therefore passed the corrected invalid-count gate (`4 <= 10`) and the reversed-view-argument invariance check, but failed both accuracy gates. Four model calls per question did not improve over B0 and significantly regressed from its M2a parent. A1 is archived and does not proceed to another source or Med-CMR.

The Agent trace contains 153 unanimous four-vote cases, 205 three-vote cases, 144 maximum-two-vote cases, six one-vote cases, and four all-invalid cases. It used the order-independent content-hash tie rule 101 times and encountered zero duplicate-content ambiguities.

## Post-gate diagnostic

No new inference was performed. Scoring the four already completed views showed:

| Cyclic shift | Accuracy | Correct | Invalid |
|---:|---:|---:|---:|
| 0 / original | 62.8906% | 322 | 10 |
| 1 | 60.1563% | 308 | 7 |
| 2 | 45.1172% | 231 | 12 |
| 3 | 45.7031% | 234 | 11 |

This exposes a stronger positional asymmetry than the original-vs-shift-1 diagnostic captured: two of the four views are substantially weaker and can outvote the useful views. Majority consensus is therefore not a safe correction. Selecting only the favorable views after observing these results would be development-set tuning and is prohibited.

PMC-512 is now analysis-only for this optimization family. Any calibrated selector, early stop, confidence weighting, or two-view policy requires a separately admitted source-diverse MCQ calibration/validation split and a frozen policy before held-out evaluation.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| rotate-2 surface report | `22e6d32bf28b4c4c0176edaa6447773e4f31624840578bdc6c082571944a29be` |
| rotate-3 surface report | `fd4a11bc7400af4d3f6cac3eb0d1a2ab3c773df9c6fc2974fb74a6667d192ac5` |
| rotate-2 run manifest | `ef4e03cddf23da16af4fb544385113a5e249f0b6426e429ccf4e7a0b29f88063` |
| rotate-2 predictions | `368696bf3e010f21b806c7c053d94c954d4b8de5bee44d0017c13d9e45bfb9e4` |
| rotate-2 metrics | `632d4bfe7765381a445f9ce10e173d8d3a96f0b4ce6d0d93320cbe40d8832334` |
| rotate-3 run manifest | `3804019272c19f20e842c3ef1694d7cc6555a2cff78fca426ba79a2b070182f6` |
| rotate-3 predictions | `5ae463be6facc50bc029b35f0f04aea2644f35699df1b53c4989536f16f790f7` |
| rotate-3 metrics | `c73b47e9a20595e518ce32a187abc63069ce7773c56079e1946577db87fe1897` |
| Agent report | `0e4633f4f7160820488f89cd6301377c0a41758a79d717bdd1f6c33431bf1636` |
| Agent predictions | `2f316c5b4d81e516896a90410d26f1a5a52a86233a3b91faccfe98388415df35` |
| Agent metrics | `ee505030a7981b6cd916a0fc2d2e68e4b00972824fae8a9c3e266c6f8a388dba` |
| comparison vs B0 | `c0bdb24257ccb1db5bc82e4058f388d996d2c67d0f489cf26af65db516ba4453` |
| comparison vs M2a | `91cadd7a45d0cf7c8534a30c42e34ef136da4d6edd0348c8f4e2eda97e2afdef` |
| frozen gate receipt | `eff27720d22137140d3ede2fd1fd5eb8ddbbbb787f6804008eb5c3e29d9ec7c4` |
