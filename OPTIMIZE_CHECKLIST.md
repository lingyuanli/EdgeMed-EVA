# Med-CMR Optimization Checklist

更新：2026-09-02

## Frontier

- current frontier mode: `explore`
- primary optimize submode: `train`
- incumbent: `qwen35-4b-medcmr-b0`, verified MCQ accuracy `27.1690%`
- promoted line: none; zero-shot B1 failed the frozen external-development comparison
- active implementation candidate: `transfer-failure-20260902`, source-diverse retention and label-invariance diagnostics
- full-eval queue: empty; Med-CMR test scoring is prohibited until the candidate and evaluation budget are frozen

## Control Checks

- [x] Durable frontier recovered from `PLAN.md`, `CHECKLIST.md`, B0 execution receipt, and verification report
- [x] Recent optimization memory reviewed
- [x] Primary submode selected: `seed`
- [x] Route selected: `exploit`
- [x] Candidate slate includes prompt/representation, training/objective, and tool/system families
- [x] Candidate briefs and shared ranking recorded in `CANDIDATE_BOARD.md`
- [x] Promotion decision limited to one line
- [x] Current implementation pool recorded
- [x] Smoke queue defined
- [x] Full-eval queue defined
- [x] B0 failures classified as capability-dominant, not parser-dominant
- [x] Stagnation check performed: no optimization attempts yet
- [x] Family-shift trigger checked: not active
- [x] Fusion eligibility checked: false; no two successful lines exist
- [x] B1-v1 implementation tests pass in the remote project environment (`22 passed`)
- [x] B1-v1 no-reference V100 smoke completes with frozen contract
- [x] B1-v1 failure classified and archived: 8/14 strict schema; do not rerun unchanged
- [x] B1-v2 repair implementation tests pass (`24 passed`)
- [x] B1-v2 no-reference V100 smoke completes with frozen contract
- [x] Smoke schema/parse/latency/memory receipt recorded
- [x] PMC-VQA v2 training seed and SLAKE validation sources frozen by immutable revision
- [x] SLAKE English validation manifest admitted: 1,053 rows, 96 images, zero confirmed Med-CMR overlap
- [x] PMC-VQA train/dev admitted: 1,968 train, 32 quarantined, 512 disjoint MCQ dev, zero confirmed Med-CMR overlap
- [x] B1-v2 external development comparison completed and hash-bound
- [x] B1-v2 archived after -17.7734 point paired answer regression despite 512/512 strict JSON
- [x] M1a two-step V100 backward/save smoke passed with finite gradients, applied optimizer steps, and hash-bound adapter
- [x] M1a saved adapter reload gate passed with 4/4 completed predictions and zero invalid parses
- [x] M1a pilot seed 20260901 improved frozen direct by +4.1016 points with paired bootstrap CI lower bound above zero
- [x] M1a pilot seeds 20260902/20260903 completed; 3/3 positive and 3/3 paired-CI lower bounds above zero
- [x] 512-step nested budget study completed; seed3/128 retained and hash-frozen
- [x] frozen best-SFT Med-CMR run completed: M1a significantly regressed by -2.8100 points despite zero invalid parses
- [x] transfer-failure campaign hypotheses, ordering, and stop rules preregistered before new GPU scoring
- [x] SLAKE proxy scorer/paired comparison implemented with answer-type/content/modality slices
- [x] deterministic answer-preserving option rotation and content-consistency scorer implemented
- [x] SLAKE direct-reasoning 64-token full launch stopped at 51 rows after answer-blind parse preflight failed (46 invalid); failed artifact preserved
- [x] bounded direct-reasoning 128-token smoke failed operational gate at 29/32 parseable; artifact preserved
- [x] first answer-only smoke isolated marker-only parser mismatch: B0 15/32 and M1a 1/32 under old parser, while failures were bare single-line answers; no references used
- [x] variant-scoped `bare_answer` parser implemented; direct Med-CMR open parsing unchanged
- [x] SLAKE answer-only B0 and M1a operational smokes each passed 32/32; hashes frozen
- [x] SLAKE full B0-vs-M1a retention slice launched after the operational gate
- [x] SLAKE B0-vs-M1a retention slice completed and hash-bound: exact +8.6420, F1 +5.5843, both paired CI lower bounds positive
- [x] H3 general semantic forgetting weakened; broad H1 same-source-only transfer weakened
- [x] PMC answer-preserving option-rotation diagnostic launched under frozen contract
- [x] PMC answer-preserving option-rotation diagnostic completed; corrected paired invariance delta -8.5938 with CI wholly negative
- [x] scorer `None -> None` overcount fixed by commit `3f95b2a`; buggy file preserved and GPU outputs unchanged
- [x] H2 answer-letter/order sensitivity supported; single-letter SFT family stopped

## Smoke Queue

1. Unit tests for prompt variant hashing, inference/reference isolation, and structured JSON parsing. `PASS: 22 tests` in the frozen remote V100 Python environment.
2. Local syntax compilation and diff checks. `PASS`; local Python 3.14 has no pytest, so behavioral tests use the project remote `.venv` rather than a new environment.
3. Remote 14-sample, two-per-task operational smoke using the existing answer-free selection and no scorer.
4. Verify 14 unique contract-bound predictions, exit code zero, schema rate, parse rate, latency, and peak memory.

## Stop And Promotion Rules

- Archive the entire zero-shot B1 structured-output line if `b1-evidence-answer-v2` cannot reach at least 13/14 parseable outputs; no second format repair is allowed.
- Do not inspect correctness or reference answers during the operational smoke.
- Do not run a full Med-CMR B1 evaluation until an external, overlap-audited development set freezes the prompt and checkpoint-selection rules.
- Promote B1 to development evaluation only if the smoke is operationally valid; smoke success alone is not evidence of accuracy gain.

## Next Concrete Action

Implement M2a semantic option-content SFT as the smallest objective change. Require original-order gain, rotated-order non-inferiority, and SLAKE retention before any new milestone. Do not reopen Med-CMR.
