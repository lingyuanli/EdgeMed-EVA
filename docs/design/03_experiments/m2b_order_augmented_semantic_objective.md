# M2b Deterministic Order-Augmented Semantic Objective

Status: `completed / invariance passed / accuracy gates failed / archived`

Parent: M2a recipe, initialized again from the frozen B0 model

Seed: `20260903`

## Why this is the next experiment

M2a significantly improved accuracy on both original and answer-preserving rotate-1 PMC development surfaces, but its 512-row content-consistency interval did not prove the frozen -1 point non-inferiority margin. Its point estimate was positive, so the smallest causal follow-up is to retain the successful option-content target while decorrelating semantic content from fixed option positions during training.

## One changed variable

M2a trains on the admitted PMC option order. M2b applies one deterministic cyclic shift to each **training** row using SHA-256 over `schema + order seed + sample_id`, and remaps the private training reference to preserve answer content. Identity shifts are retained as part of the frozen uniform hash mapping.

Everything else remains M2a-identical: B0 initialization, 1,968 admitted rows, the seed-20260903 256-example selection, semantic option-text target, image preprocessing, QLoRA rank/alpha/dropout and trainable modules, optimizer, learning rate, accumulation, 128 optimizer steps, FP16/NF4 V100 runtime, inference prompt, parser, scorer, and original/rotate-1 development surfaces.

The transform does not touch evaluation data. It writes an answer-free inference surface, a mode-0600 reference file, source/output hashes, all-row answer-position counts, shift counts, and the exact 256-example training-selection audit. Frozen B0 semantic predictions are reused only after their hashes match the M2a archive.

## Preregistered gates

1. Surface gate before GPU work:
   - 1,968 unique output rows and exact input ID equality;
   - answer content preserved for every row;
   - no reference fields in the inference surface;
   - all four cyclic shifts represented;
   - each answer position is between 20% and 30% of the frozen 256-example selection.
2. Two-step V100 smoke: finite loss/gradients, two applied optimizer steps, saved adapter hashes, peak memory, and four completed reload predictions.
3. One 128-step seed only; no M2a rerun and no seed search.
4. Frozen PMC gate, conjunctive:
   - original-order M2b-minus-B0 accuracy is positive;
   - rotated-order paired 95% CI lower bound is at least -1 point;
   - content-consistency paired 95% CI lower bound is at least -1 point.
5. Only if all PMC checks pass: run the already frozen SLAKE answer-only retention comparison.

Failure archives M2b without changing its order seed or inspecting per-sample correctness. No M2a/M2b result authorizes Med-CMR access; a new milestone would require PMC plus SLAKE gates and a separately frozen test decision.

## Surface and smoke receipt

The frozen transformation produced 1,968 answer-preserving rows. The original answer-position counts `241/712/769/246` became `519/510/463/476`; all-row shift counts were `476/475/514/503`. The exact 256 training examples selected by the unchanged seed have A/B/C/D counts `66/71/58/61`, satisfying the preregistered 20%–30% bounds. Their ordered sample-ID list SHA-256 is `11380adbb4bd820e3c4d23d3ca6ad3ef7eb94d3787efa6ffa17d1b55316164e8`.

The V100 smoke exited zero with 2/2 optimizer steps applied, finite loss (first `0.832290`, last `0.122301`, mean `0.484738`), peak CUDA allocation `6903.48 MiB`, a saved adapter, and 4/4 reload outputs parsed as `option_text_match`.

| Artifact | SHA-256 |
|---|---|
| transformation report | `fa7d1b8482dea3884a65623607f060b569887db4c4a05f35a543ba37f87db8fa` |
| smoke training manifest | `c7d40f5fda1daa24e3e4f2cb7f6442ffbe69c1fb09fc3077d13e1146e1c812ea` |
| smoke adapter model | `5fb64ab76cf02394f611770219d781babc148e2d25f100f3505e9f22b551da4a` |
| reload run manifest | `81bb989a59724d1566ae56b451798562eb93b4fcc22e1e9b28f415e0b0ed23cf` |
| reload predictions | `75a1d00f6ff1205bdaeeeceab9bc10ce77a024f68fafed9307fa325803b1f15f` |

The single 128-step pilot was launched in remote tmux session `m2b-orderaug-pilot128` from code commit `30ec7e0e3f6f5a013f8a7b9a0c757f73dc411577`. The launch preflight matched the frozen B0 original/rotate-1 prediction hashes and all four development-surface hashes; no B0 inference is repeated.

## Frozen pilot result

The run exited zero with 512/512 unique predictions on both evaluation views. M2b accuracy was 56.8359% original and 52.5391% rotate-1, respectively `-1.1719` and `-3.1250` points versus B0. Their paired intervals `[-5.4688, 3.1250]` and `[-7.4219, 0.9766]` fail the accuracy gates.

Content consistency improved from B0 70.5078% to M2b 79.1016%, a significant `+8.5938` points with interval `[3.9063, 13.4766]` and exact McNemar `p=0.000736`. Thus deterministic training-order augmentation changed the intended mechanism but incurred an accuracy trade-off. The overall conjunctive gate is failed; M2b is archived without SLAKE or Med-CMR access.
