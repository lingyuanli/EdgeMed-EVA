# Qwen3.5-4B Med-CMR B0 Verification

## Verdict

- MCQ: `verified_diverged`
- feasibility: `PASS` on one Tesla V100-SXM2-32GB
- downstream trust: suitable as the frozen, untrained B0 starting point; not evidence that the target small model beats commercial baselines
- Open: `operational_but_incomparable` because the paper's exact DeepSeek-V3.2-Exp judge is unavailable

The MCQ execution is complete and internally reproducible, but its 27.1690% accuracy is far below the paper's reported strong baselines. It therefore establishes the optimization starting line rather than the claimed end result.

## Frozen Evidence

- run: `/home/ubuntu/EdgeMed-EVA/runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z`
- execution commit: `f26fa980572044bf843da303ab930f2e2f520938`
- run contract: `7de0a22edcef5c4ad084e9d57c74d306be99048c9234c36946f13cbbc030a96a`
- manifest: `9ec6f833f1f53509d25873b2beb77960f18d55b4b514a0f4796efd147d0219d7`
- references: `b9ae4474d767599f970593deaee4481d452a712278f4874bcd084aadc3f773e6`
- predictions: `3c6c32bc5254d7145ff6f06e81f09cac437df8bdd2ca5712de3ef683bb248719`
- metrics: `c3c027001f7419d18402da1943a4a7874af86e812c0ee9e0c295853578ad634d`
- process exit code: `0`
- run status: `completed`, `16,655/16,655`, `resume_count=0`

## Metric Result

| Slice | Correct / Total | Accuracy |
|---|---:|---:|
| Overall | 4,525 / 16,655 | 27.1690% |
| FDD | 298 / 711 | 41.9128% |
| TP | 106 / 336 | 31.5476% |
| CR | 603 / 1,922 | 31.3736% |
| MSI | 218 / 709 | 30.7475% |
| SOD | 310 / 1,024 | 30.2734% |
| SU | 112 / 371 | 30.1887% |
| LTG | 2,878 / 11,582 | 24.8489% |

Overall Wilson 95% CI is `26.4988%–27.8498%`. The parser produced 14,288 exact letters, 1,588 leading option labels, 5 standalone-line answers, 1 answer-marker answer, and 773 invalid outputs. Invalid outputs were retained and counted as incorrect.

## Independent Verification

The verification implementation did not import `edgemed_bench.score_mcq`. It loaded the manifest, references, and predictions independently and checked:

1. all three inputs contain exactly 16,655 rows and 16,655 unique `sample_id` values;
2. prediction and reference ID sets exactly equal the manifest ID set, and prediction order equals manifest order;
3. every prediction is bound to the frozen run-contract hash;
4. actual manifest and prediction hashes equal the hashes declared by the run manifest;
5. direct `parsed_answer == reference.answer` comparison reproduces `4,525/16,655` and every task-slice numerator/denominator exactly;
6. independently counted null answers reproduce the scorer's 773 invalid parses;
7. the scorer's source hashes equal fresh SHA-256 calculations.

All checks passed on `2026-09-01T13:56+08:00`. No answer or label field is present in the prediction schema; reference answers stayed in the scorer-only file.

## Comparability Boundary

This result uses the official source-pinned released MCQ set, deterministic generation, and a documented parser. It is best described as an official-data, independently verified reproduction result. It is not an official leaderboard score because the official evaluator implementation and exact regex extraction rule are not published. Open-ended results remain outside the verified baseline until the exact judge contract is available or an explicitly labeled proxy protocol is adopted.
