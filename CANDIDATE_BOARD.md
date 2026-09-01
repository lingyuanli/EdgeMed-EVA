# Med-CMR Candidate Board

更新：2026-09-01

## Shared Ranking Surface

Candidates are ranked by information gain, feasibility on one V100, test-comparability safety, implementation scope, expected incumbent improvement, mechanism distinctness, and failure risk. B0 model weights, official data revision, image preprocessing, deterministic decoding, scorer, and source binding remain unchanged unless a later candidate explicitly changes one of them.

| Rank | Candidate ID | Level | Parent | Strategy | Mechanism / Layer | Status | Expected Gain | Observed Result | Promote / Archive |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | `b1-structured-json-v1` | implementation | `qwen35-4b-medcmr-b0` | exploit | prompt + representation / Tier1 | `proposed` | Cheaply tests whether explicit observation, competing hypotheses, and strict answer structure improve schema behavior before training | n/a | **promote to smoke now** |
| 2 | `m1b-evidence-sft` | brief | `qwen35-4b-medcmr-b0` | explore | supervised objective / Tier2 | `held_data_gate` | Largest plausible capability gain, especially on LTG, while enabling evidence metrics | n/a | hold until 2k clean seed + golden validation pass provenance/overlap QA |
| 3 | `b2-selective-crop` | brief | `qwen35-4b-medcmr-b0` | explore | tool/system / Tier2 | `held_oracle_gate` | May improve SOD/FDD if native resolution is limiting | n/a | hold until an answer-blind oracle-crop study shows useful upper bound |

## Winning Brief: B1 Structured Evidence

- bottleneck: B0 is capability-limited and emits 4.64% invalid answers; no evidence trace exists to separate observation, reasoning, and answer failures.
- why current line is limited: direct answer output exposes only correctness and cannot measure whether the model observed relevant visual evidence or merely guessed from option priors.
- mechanism: request one compact JSON object containing a factual visual observation, 1–3 competing option-letter hypotheses, and one final A–E answer; parse and retain these fields deterministically.
- mechanism family: `prompt + representation`.
- change layer: `Tier1`.
- source lens: `baseline_refinement`.
- why now: B0 is fully verified, so a prompt-only delta is cheap, attributable, and executable on the current V100.
- keep unchanged: Qwen3.5-4B revision, NF4/FP16/eager runtime, images, preprocessing, deterministic decoding, MCQ scorer, and answer isolation.
- expected gain: first establish structured-output feasibility and expose evidence failure modes; accuracy gain is not assumed.
- implementation surface: `prompts.py`, `parsing.py`, `run.py`, focused tests, and one remote smoke receipt.
- main risks: verbose output truncation, cosmetic evidence unrelated to the answer, or prompt-induced accuracy regression.
- disconfirmation: operational parse rate below 13/14 after at most one evidence-backed repair, or later clean-development accuracy materially below direct B0.
- promote now: yes, to bounded no-reference smoke only.
- next target: `experiment` after smoke contract freezes.

## Non-Winner Notes

- `m1b-evidence-sft` remains the likely main performance line, but starting it before clean training/validation data would make the result untrustworthy.
- `b2-selective-crop` is deliberately deferred because B0's dominant LTG weakness is not evidence that crop is useful; tool work requires an oracle upper bound first.
- promotion cap is one. There is no fusion candidate because no optimized line has measured strength yet.
