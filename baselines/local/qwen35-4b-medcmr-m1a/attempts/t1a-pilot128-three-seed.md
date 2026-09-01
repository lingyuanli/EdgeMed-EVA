# T1a Pilot-128 Three-seed Development Receipt

Date: 2026-09-01  
Tier: auxiliary development; **not an official Med-CMR score**  
Verdict: three-seed pilot passed; promote M1a to a bounded training-budget study

All seeds used the same 128 optimizer-step budget, 256 admitted training
examples, answer-token-only objective, direct prompt, rank-16 language-only
LoRA, frozen vision/projector, NF4/FP16, image cap, scorer, 512-row development
set, and frozen untrained direct predictions. Only the training seed changed.

| Seed | Accuracy | Delta vs direct | Paired bootstrap 95% CI | McNemar p | Invalid |
|---:|---:|---:|---:|---:|---:|
| 20260901 | 61.7188% | +4.1016 | [0.3906, 8.0078] | 0.0438753 | 0/512 |
| 20260902 | 63.8672% | +6.2500 | [2.1484, 10.3516] | 0.00380693 | 0/512 |
| 20260903 | 66.0156% | +8.3984 | [4.2969, 12.5000] | 0.000100579 | 0/512 |

- Frozen direct: 295/512, 57.6172%.
- Three-seed mean accuracy: 63.8672%; sample standard deviation: 2.1484 points;
  range: [61.7188%, 66.0156%].
- Mean delta: +6.2500 points; sample standard deviation: 2.1484 points;
  range: [+4.1016, +8.3984].
- Direction: 3/3 positive; paired-CI lower bound: 3/3 positive.
- All training runs completed 128/128 finite/applied optimizer steps.
- Peak allocated training memory ranged from 8,051.89 to 8,691.91 MiB.

## Immutable seed bindings

| Seed | Train contract | Train manifest | Adapter | Eval contract | Predictions | Paired report |
|---:|---|---|---|---|---|---|
| 20260901 | `aa9b7315cdf0ace466be7ff83017829c716af0d11e377401e54486453d720897` | `56d7e314a7b0b0e188a7d040e0fce20ecdd0adfef49e4e19c27b7d2eafb9072f` | `60899bbc3d3aff6d3724e2414c008f8f556af7c5df5ea8c327b51cf4606cbd66` | `7d8920ef33e5ff0b1466150f9eeee52fe08927660eb9f46ec5f217a0c2751fd2` | `20210089a21f9f197cef132dcf7e0849a056b5052613f3586869cdd960028ed9` | `2e2f75fd946e90b5ce7018b2be9ac2abf322662d5b6783431c0c17613afab027` |
| 20260902 | `f3861da6f9fe6faf15956e0b8ec7ddd24f4d9b488a7032834fac8b134aede6c0` | `5e47b428bb116c09f73b94f536c3f1f11daa5f8795eb38378b7383199f701efe` | `0b87b9eb0e1574c3f5aa21305181e9abb428abf44e3fd60084f7d5b793bbbcbd` | `58f30d96f389b71af66036175fa78cbb2fe4a5118cb535ce9c28b6127ea6ceff` | `4d4aa730f3d38949b35d55aed851f8e28b08cff746ebc71a2c3f7192b133edae` | `33bc4f20f2944db004f3bfc99fdd93f68ce371bccbeef00ca08dbe2123727b5e` |
| 20260903 | `6a6fac039208c7e2c86df2263f4143f55b0c8a39b23b191676ccd8cadfb3ab8f` | `ee96427dbd922ecf4f192fed3caae3b94c880dccd7475243e39d2a301b0d79b0` | `874233467cae2428524a5184d702667e71ab2e6c49bccc41712fd58b87b9c64c` | `03c295c8fb790b90e68aa276eb422155d38b662ce0e8f72306ad503d416e5c87` | `1de2e82e03b2996d304bbc8dc19a84223413766018d5b5c57de2f719d8437b4b` | `8350da783b0a1d03a50053dbc805fb96baccf1a98abe4d53032e9001c5ddf14e` |

The mechanism is robust enough to continue on external development data, but
the result does not authorize a claim of Med-CMR improvement. The next single
change is training budget: rerun seed 20260903 from initialization for 512
steps, preserving every other contract field. Retain the 128-step seed3 adapter
unless the longer run improves the same paired development gate.
