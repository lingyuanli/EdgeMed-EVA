# Qwen3.5-4B Med-CMR M1a Execution

## Frozen candidate

- milestone: best supervised fine-tuning checkpoint
- checkpoint: PMC-VQA T1a seed 20260903, 128 optimizer steps
- adapter SHA-256: `874233467cae2428524a5184d702667e71ab2e6c49bccc41712fd58b87b9c64c`
- training manifest SHA-256: `ee96427dbd922ecf4f192fed3caae3b94c880dccd7475243e39d2a301b0d79b0`
- training contract SHA-256: `6a6fac039208c7e2c86df2263f4143f55b0c8a39b23b191676ccd8cadfb3ab8f`
- selection evidence: three 128-step seeds were positive on external development;
  seed3 was best; a nested 512-step run did not improve it.
- prompt: unchanged B0 direct MCQ prompt
- inference runtime: unchanged B0 NF4/FP16/eager deterministic runner
- evaluation budget: one frozen-best-SFT full Med-CMR MCQ run
- state: `completed / verified regression / archived for answer optimization`
- remote run directory: `/home/ubuntu/EdgeMed-EVA/runs/qwen35-4b-medcmr-m1a-sft128-s20260903-mcq-full-20260901`
- tmux session: `medcmr-m1a-full`
- code commit: `f5f00837d3b413b3ebcf53aff730e68cd7ccf618`
- run contract SHA-256: `1dc588abbc18855eb3d99fa8df2b375195ad86b18c4b8fe21c434d65206f93b4`
- launch evidence: predictions increased through 16 rows while the Python GPU
  process held approximately 4.24 GiB; process exit remains pending.
- expected count: 16,655; no aggregate metric is read before completion.

## Completion

- process exit code: `0`
- coverage: 16,655/16,655 unique predictions; no resume
- finished at: `2026-09-01T13:27:08.851007+00:00`
- inference time: 10,501.56 s (2 h 55 min 1.56 s), plus 7.52 s model load
- peak allocated CUDA memory: 3,854.03 MiB
- run-manifest SHA-256: `e571a3702caf9aff428a7786727c53607de208430dcba51081c20612c532220d`
- predictions SHA-256: `ad7b61d23aa7a6e578b82d1df8e102f16b42eadb8adc7f43b0aafcf2ce3a59d5`
- metrics SHA-256: `396415a6437ed5d5f298f8fd481b003d0b06e4c82c23ca673bc082642fd6ec57`
- paired-report SHA-256: `30a7d8c603601bf5b6fc48348e4a3e40cce8716b0b9bee4e272842c853a4e19a`

## Verified result

| Metric | B0 | M1a | M1a minus B0 |
|---|---:|---:|---:|
| Overall | 27.1690% | 24.3591% | **-2.8100** |
| SOD | 30.2734% | 29.9805% | -0.2930 |
| FDD | 41.9128% | 37.6934% | -4.2194 |
| SU | 30.1887% | 29.1105% | -1.0782 |
| TP | 31.5476% | 29.1667% | -2.3810 |
| CR | 31.3736% | 27.9396% | -3.4339 |
| LTG | 24.8489% | 22.0428% | -2.8061 |
| MSI | 30.7475% | 26.2341% | -4.5134 |

- M1a correct: 4,057/16,655; Wilson 95% CI `[23.7131%, 25.0168%]`.
- Invalid parses: 0/16,655, down from B0's 773/16,655.
- Paired bootstrap 95% interval for M1a minus B0:
  `[-3.3443, -2.2696]` accuracy points.
- Exact two-sided McNemar p-value: `2.3391046176276942e-24`.
- Contingency: both correct 3,231; B0-only correct 1,294; M1a-only correct
  826; both wrong 11,304.

M1a therefore fixes answer-format compliance but significantly reduces answer
accuracy, including regression in all seven Med-CMR dimensions. The strong
PMC-VQA development gain did not transfer to Med-CMR. This completed run
consumes the best-SFT test milestone and must not be used for per-sample tuning,
checkpoint reselection, or repeated M1a test attempts.

The full run may expose aggregate Med-CMR metrics only after all 16,655
predictions complete. Per-sample test correctness is not used for further
prompt, parser, crop, hyperparameter, or checkpoint selection.
