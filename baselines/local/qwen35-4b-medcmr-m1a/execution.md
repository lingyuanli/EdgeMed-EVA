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
- state: `frozen / launch pending`

The full run may expose aggregate Med-CMR metrics only after all 16,655
predictions complete. Per-sample test correctness is not used for further
prompt, parser, crop, hyperparameter, or checkpoint selection.
