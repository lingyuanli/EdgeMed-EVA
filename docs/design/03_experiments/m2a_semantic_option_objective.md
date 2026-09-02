# M2a Semantic Option-Content Objective

Status: `implementation / smoke next`  
Parent: B0 model and M1a training recipe, not M1a weights

## Confirmed cause addressed

M1a's single-letter objective amplified answer-order sensitivity: its content consistency under a deterministic answer-preserving rotation was 63.2813% versus B0 71.8750%, paired delta -8.5938 with a confidence interval wholly below zero. At the same time, M1a improved human cross-source SLAKE exact/F1, so the narrow repair is to retain domain supervision while removing the answer-letter output target.

## Single changed variable

- M1a target: one assistant letter token such as `B`.
- M2a target: `Answer: <complete text of option B>`.

The model, source rows, seed, order, image budget, QLoRA layers/rank/alpha/dropout, optimizer, learning rate, accumulation, step count, FP16/NF4 runtime, frozen vision/projector, and finite-gradient gates remain unchanged.

At inference, `semantic_option` asks for the complete text of one visible option. The parser normalizes the generated string and visible option strings and accepts only a unique exact match. It maps that visible content back to a letter for the existing scorer. It never reads the reference answer.

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

