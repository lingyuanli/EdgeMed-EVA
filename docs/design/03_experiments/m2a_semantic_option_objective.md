# M2a Semantic Option-Content Objective

Status: `completed / accuracy gates passed / invariance gate inconclusive / not promoted`
Parent: B0 model and M1a training recipe, not M1a weights

## Confirmed cause addressed

M1a's single-letter objective amplified answer-order sensitivity: its content consistency under a deterministic answer-preserving rotation was 63.2813% versus B0 71.8750%, paired delta -8.5938 with a confidence interval wholly below zero. At the same time, M1a improved human cross-source SLAKE exact/F1, so the narrow repair is to retain domain supervision while removing the answer-letter output target.

## Single changed variable

- M1a target: one assistant letter token such as `B`.
- M2a target: `Answer: <complete text of option B>`.

The model, source rows, seed, order, image budget, QLoRA layers/rank/alpha/dropout, optimizer, learning rate, accumulation, step count, FP16/NF4 runtime, frozen vision/projector, and finite-gradient gates remain unchanged.

At inference, `semantic_option` asks for the complete text of one visible option. The parser normalizes the generated string and visible option strings and accepts only a unique exact match. It maps that visible content back to a letter for the existing scorer. It never reads the reference answer.

The B0 preflight additionally exposed `Answer: D) <exact option text>`. Parser v2 accepts this only when the stated label and following text identify the same visible option; a label alone or a mismatched label/text pair remains invalid. The partial parser-v1 run stopped after 66 rows and was preserved with predictions SHA-256 `bdaf1a0…68c20`. The completed 128-step adapter is reused; training is not repeated.

Parser-v2 answer-blind smoke passed: B0 30/32 parseable (`30 option_label_text_match`, two label-only invalid), M2a 31/32 (`31 option_text_match`, one invalid). Prediction hashes are `9727212e…f2584` and `975c0efa…974f`. Full evaluation reuses the completed adapter with `EDGEMED_SKIP_TRAIN=1`.

## Gates

1. Two-step V100 smoke: finite loss/gradients, two applied optimizer steps, adapter hashes, peak memory, and 4-row reload inference.
2. One 128-step seed (`20260903`) on the existing PMC training surface.
3. Same-prompt B0 versus M2a on original PMC dev and the frozen rotate-1 surface.
4. M2a must not reproduce the confirmed shortcut:
   - original-order candidate gain is positive;
   - rotated-order paired CI lower bound is at least -1 point versus B0;
   - content-consistency paired CI lower bound is at least -1 point versus B0.
5. Only after PMC gates pass: SLAKE exact/F1 retention versus the frozen B0 answer-only run.

Failure at gates 2–4 archives plain semantic targets and promotes deterministic order augmentation as a separately preregistered variable. No result authorizes Med-CMR access.

## Smoke receipt

- exit code: 0
- optimizer steps: 2/2 applied
- loss: finite; first `0.677792`, last `0.125183`
- peak CUDA memory: `6903.48 MiB`
- adapter model SHA-256: `945cf789f831d179f1e628d556858291d769358698c71e85a83349e85ce30bf2`
- reload inference: 4/4 completed, all `option_text_match`
- code commit: `7705ee6b0cbe2cefc06a79cc6a81d9c245e4b020`

The full 128-step training completed with finite mean loss `0.183526`, last loss `0.147254`, and peak CUDA allocation `8475.42 MiB`. Its evaluation was restarted under parser v2 without retraining.

## Frozen pilot result

All original/rotated B0 and M2a runs completed 512/512 with zero invalid parses. M2a improved original-order accuracy by `+4.8828` points (95% CI `[0.7813, 8.9893]`, `p=0.0261`) and rotated-order accuracy by `+4.4922` points (95% CI `[0.5859, 8.3984]`, `p=0.0346`). This is a substantial repair over M1a's rotation-sensitive result.

Content consistency was 70.5078% for B0 and 72.2656% for M2a. The paired delta was `+1.7578` points, but its interval `[-3.1250, 6.6406]` did not establish the preregistered -1 point non-inferiority margin. Because the gate was conjunctive, the overall receipt is `failed`; this means evidence is insufficient for promotion, not that the accuracy repair failed.

M2a does not proceed to SLAKE or Med-CMR. The next isolated change is M2b deterministic training-only option-order augmentation. Full receipts and hashes are archived under `baselines/local/qwen35-4b-medcmr-m2a/attempts/`.
