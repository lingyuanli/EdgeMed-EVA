# M2b Order-Augmented Semantic Pilot, seed 20260903

Status: `completed / invariance passed / accuracy gates failed / archived`

Date: 2026-09-02

Parent recipe: M2a semantic option-content SFT, initialized from frozen B0 weights

Single changed variable: deterministic per-sample cyclic option shifts on the training surface

## Result

Both M2b evaluation runs completed 512/512 unique rows. Five outputs on each surface were unparseable and were counted as incorrect.

| Metric | B0 | M2b | M2b - B0 | Paired 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| original-order accuracy | 58.0078% | 56.8359% | -1.1719 | [-5.4688, 3.1250] | 0.6587 |
| rotate-1 accuracy | 55.6641% | 52.5391% | -3.1250 | [-7.4219, 0.9766] | 0.1706 |
| content consistency | 70.5078% | 79.1016% | +8.5938 | [3.9063, 13.4766] | 0.000736 |

## Gate decision

- `FAIL`: original-order delta is not positive.
- `FAIL`: rotated-order interval lower bound is below -1 point.
- `PASS`: consistency interval lower bound is above -1 point and wholly positive.
- overall conjunctive gate: `failed`.

M2b supplies strong causal evidence that training-order randomization can improve option-order consistency, but it trades away answer correctness on both evaluation views. It is archived and does not proceed to SLAKE or Med-CMR. The order seed is not tuned, no extra M2b seed is run, and no per-sample correctness inspection is used.

The next family shifts from training to Agent inference: keep the more accurate frozen M2a weights and aggregate complete-option-text predictions across a preregistered set of answer-preserving cyclic views. This directly targets the observed instability without teaching away the source distribution.

## Training receipt

- optimizer steps: 128/128 applied
- examples seen: 256
- loss: finite; first `0.832290`, last `0.046613`, mean `0.187550`
- peak CUDA allocation: `8475.42 MiB`
- training manifest SHA-256: `2105562e64c187a95d1c18b313c3475001e09c2fce0d156b80d54f47f1da6a37`
- adapter model SHA-256: `36fd77590ffe7bf4408a49658ee2506b1cc84b1ab0db0922525cd449b765361b`
- code commit: `30ec7e0e3f6f5a013f8a7b9a0c757f73dc411577`

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| original run manifest | `7ce97f71bf6b5fadd828e85a01a98d49fe0c54fb0ad075236aa7b8e95d9706c3` |
| original predictions | `d510edd146914f90db337166a8645545e519f59836a8c81c7341023ea5ec75be` |
| original corrected metrics | `7bc03f754761ef5e5518a9ff0fb6fa3a1c17edc74b7e0f31a86eeb6f834bf00b` |
| original pre-fix metrics | `627916cb0b7b33497a81e155988876f8112224ea5bc988531fc027b9c355907b` |
| rotate-1 run manifest | `21f7ad24216e45a261bfa00329d546add8aac6abcb3baed64d14ae6959d834d2` |
| rotate-1 predictions | `43dc312e4c106675f060d2b92df55786754f3c84ad5c798328ac10ed6dcb94d0` |
| rotate-1 corrected metrics | `eabf684de0bb4adbd139cdacb2764bf023e6c584310230cdf3277cfa9fee20a9` |
| rotate-1 pre-fix metrics | `f620e82f110691330dbd00699b39f278a5a151afece0499ae26903376a614c7c` |
| M2b consistency | `424415f7745aa785f28c6fefa55d2b9ee7bc1fc7589ff93d2a185a2033c60e72` |
| original paired comparison | `53f9577ddbf9855c98c27055dc67c1fa40d6b922fb5b0f64e99d9c64f6096c73` |
| rotate-1 paired comparison | `84c7b0a09fb123ac37fe6466f933d7adc724f1e653bc815d12d3bcc5275da8a4` |
| consistency paired comparison | `0b869005fe23f6b4805046eef0f487fe4e9d45924c677b9c3c2cbeb1a19aeb4e` |
| frozen gate receipt | `66b406eedb94d34aa7ff377c2ab8162c77a3e161fbec6789739633bb8c5c48e3` |

## Scorer repair note

The run used the older aggregate scorer, which counted only a literal `parse_status=invalid`. Commit `a5ddf38` instead counts `parsed_answer is None`, covering `invalid_option_text`. Old metric files were preserved as `metrics.buggy-invalid-count-v1.json`; only aggregate invalid fields and their file hashes changed. Accuracy and paired statistics were unchanged.
