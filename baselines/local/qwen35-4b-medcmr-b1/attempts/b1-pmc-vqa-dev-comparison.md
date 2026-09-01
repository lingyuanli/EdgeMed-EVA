# B1 PMC-VQA Development Comparison Receipt

Date: 2026-09-01  
Tier: auxiliary development; **not an official Med-CMR score**  
Decision: archive zero-shot `b1-evidence-answer-v2` for answer optimization

## Frozen comparison

- Data: 512 admitted PMC-VQA v2 MCQs from the official test source.
- Inference surface SHA-256: `78c6d2a5c0790eaf3c66db523774f4cfcfeb93e27d4cf743e540f8bbfdff5e75`.
- Reference surface SHA-256: `9a7e03cfdfa258b02eecace8c805dfceeeb247ee12cc63b635aeb003b4b3b0f6`.
- Model/runtime commit: `cb29337f1d789b2a6033986bf311535fd45d7a08`.
- Model, quantization, image preprocessing, deterministic decoding, samples, and
  scorer were identical. The changed surface was the prompt/output schema only.
- Both runs completed 512 unique predictions with exit code zero.

| Arm | Contract SHA-256 | Predictions SHA-256 | Correct | Accuracy | Invalid/schema |
|---|---|---|---:|---:|---:|
| direct (A) | `e38e89c0c1207cd54c4f95039e60d8faf994d7d37939111919d06bfd69a5d0e2` | `b9c8eb8af8fc156253a4a0d4f9bc50572a4c52fa04c457ba796e7c1deeeba518` | 295/512 | 57.6172% | 4 invalid answers |
| evidence + answer v2 (B) | `876c5c889915f671325b6fbd747851e1506143971d34d995419cd9a0c29281d2` | `dd93fd7c0379bdfb752e774a280bf26881e3cac5c39680855987e3385c82e450` | 204/512 | 39.8438% | 512/512 strict JSON |

## Paired result

- B minus A: **-17.7734 accuracy points**.
- 10,000-repetition paired-bootstrap 95% interval: **[-22.8516, -12.5000]**.
- Exact two-sided McNemar p-value: **1.0477413530388188e-10**.
- Contingency: both correct 149; A-only correct 146; B-only correct 55;
  both wrong 162.

The structured prompt improves the observable output contract but causes a
large, statistically clear answer regression. Strict JSON compliance is not
evidence that the observation is medically grounded. B1 therefore fails the
development promotion rule and must not be described as an accuracy
improvement or sent to another Med-CMR test evaluation.

## Resource evidence

- Direct: 366.63 inference seconds; 4,435.93 MiB peak allocated CUDA memory.
- B1: 1,056.74 inference seconds; 4,436.29 MiB peak allocated CUDA memory.
- Both used one Tesla V100-SXM2-32GB with the uniform 786,432-pixel cap.

## Next decision

Proceed to the independently motivated `m1a-answer-qlora` backward smoke using
the direct answer objective. Do not train the failed B1 observation schema and
do not use PMC captions as evidence targets.
