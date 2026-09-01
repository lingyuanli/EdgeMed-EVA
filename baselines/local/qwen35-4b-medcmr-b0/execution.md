# Qwen3.5-4B Med-CMR B0 Execution

## Full MCQ Run

- state: `completed` and independently verified
- remote host: `ubuntu@117.50.188.27`
- tmux session: `medcmr-b0-mcq-full`
- run directory: `/home/ubuntu/EdgeMed-EVA/runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z`
- start time: `2026-08-31T05:55:30.702509+00:00`
- code commit: `f26fa980572044bf843da303ab930f2e2f520938`
- run contract SHA-256: `7de0a22edcef5c4ad084e9d57c74d306be99048c9234c36946f13cbbc030a96a`
- dataset manifest SHA-256: `9ec6f833f1f53509d25873b2beb77960f18d55b4b514a0f4796efd147d0219d7`
- selected sample count: `16655`
- selected IDs SHA-256: `43da0bee65a0ec53e04c7c361e818a92b95721229dac1612eddcaeb18160721a`
- model source manifest SHA-256: `a8bfc09b80581bd5d74065ca9574da513433e46b6ed117bad19c5858f2d03def`
- prompt SHA-256: `d8245cf6e33e209b9819935c40bea5a9a47efcd5c8469842e87afb2d18a9160a`
- launch script: `scripts/run-medcmr-b0-mcq-full.sh`
- finish time: `2026-08-31T09:24:29.859863+00:00`
- inference time: `12,532.83 s` (3 h 28 min 52.83 s), plus `6.31 s` model load
- process exit code: `0`
- completed samples: `16,655/16,655`, all sample IDs unique and exactly equal to the manifest
- predictions SHA-256: `3c6c32bc5254d7145ff6f06e81f09cac437df8bdd2ca5712de3ef683bb248719`
- metrics SHA-256: `c3c027001f7419d18402da1943a4a7874af86e812c0ee9e0c295853578ad634d`

First live evidence after launch: Python PID `25835`, GPU allocation about 4.26 GiB, and predictions increased from 47 to 61 while the console log advanced. The run completed without resume (`resume_count=0`); maximum allocated GPU memory recorded by the runner was 3,730.18 MiB.

## Completion Verification

All completion gates passed. The scorer ran without `--allow-incomplete`:

```bash
cd /home/ubuntu/EdgeMed-EVA
PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
  --manifest /home/ubuntu/data/medcmr/release_a9b2d6e6/manifests/mcq.jsonl \
  --references /home/ubuntu/data/medcmr/release_a9b2d6e6/references/mcq.jsonl \
  --predictions runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z/predictions.jsonl \
  --output runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z/metrics.json
```

## Verified MCQ Result

- overall accuracy: **27.1690%** (`4,525/16,655`)
- Wilson 95% CI: `26.4988%–27.8498%`
- missing samples: `0`
- invalid parse: `773/16,655` (`4.6412%`), counted as incorrect
- FDD: `41.9128%` (`298/711`)
- TP: `31.5476%` (`106/336`)
- CR: `31.3736%` (`603/1,922`)
- MSI: `30.7475%` (`218/709`)
- SOD: `30.2734%` (`310/1,024`)
- SU: `30.1887%` (`112/371`)
- LTG: `24.8489%` (`2,878/11,582`)

This is the verified B0 result for the repository's source-pinned MCQ reproduction, not an official leaderboard submission. The official repository does not publish evaluator code and the paper does not disclose its exact answer-extraction regex. See `verification.md` for the independent recomputation and comparability boundary.

Open-ended generation and judge scoring were not part of this MCQ process. The paper's exact `DeepSeek-V3.2-Exp` judge is currently unavailable, so the open component is `operational_but_incomparable` and no full Med-CMR score is claimed.
