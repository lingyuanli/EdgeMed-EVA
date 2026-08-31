# Qwen3.5-4B Med-CMR B0 Execution

## Full MCQ Run

- state: `running` (not a score and not complete)
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

First live evidence after launch: Python PID `25835`, GPU allocation about 4.26 GiB, and predictions increased from 47 to 61 while the console log advanced. PID is observational and may change only after an explicitly authorized exact resume; hashes and run directory are authoritative.

## Completion Verification

Do not report an MCQ score until the run manifest is `completed`, predictions contain exactly 16,655 unique contract-bound samples, `process_exit_code` is zero, and the independent scorer succeeds without `--allow-incomplete`.

```bash
cd /home/ubuntu/EdgeMed-EVA
PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
  --manifest /home/ubuntu/data/medcmr/release_a9b2d6e6/manifests/mcq.jsonl \
  --references /home/ubuntu/data/medcmr/release_a9b2d6e6/references/mcq.jsonl \
  --predictions runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z/predictions.jsonl \
  --output runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z/metrics.json
```

Open-ended generation and judge scoring are not part of this running MCQ process. The paper's exact `DeepSeek-V3.2-Exp` judge is currently unavailable, so no open-ended result may be labeled official-equivalent yet.
