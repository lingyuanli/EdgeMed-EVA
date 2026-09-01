# Med-CMR Optimization Checklist

更新：2026-09-01

## Frontier

- current frontier mode: `exploit`
- primary optimize submode: `seed`
- incumbent: `qwen35-4b-medcmr-b0`, verified MCQ accuracy `27.1690%`
- promoted line: `B1 structured evidence`, prompt-only diagnostic with unchanged model/data/image preprocessing/decoding
- active implementation candidate: `b1-structured-json-v1`
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
- [ ] B1 implementation tests pass locally
- [ ] B1 no-reference V100 smoke completes with frozen contract
- [ ] Smoke schema/parse/latency/memory receipt recorded

## Smoke Queue

1. Unit tests for prompt variant hashing, inference/reference isolation, and structured JSON parsing. `PASS: 22 tests` in the frozen remote V100 Python environment.
2. Local syntax compilation and diff checks. `PASS`; local Python 3.14 has no pytest, so behavioral tests use the project remote `.venv` rather than a new environment.
3. Remote 14-sample, two-per-task operational smoke using the existing answer-free selection and no scorer.
4. Verify 14 unique contract-bound predictions, exit code zero, schema rate, parse rate, latency, and peak memory.

## Stop And Promotion Rules

- Archive `b1-structured-json-v1` if it cannot reach at least 13/14 parseable outputs after one root-cause-confirmed format repair.
- Do not inspect correctness or reference answers during the operational smoke.
- Do not run a full Med-CMR B1 evaluation until an external, overlap-audited development set freezes the prompt and checkpoint-selection rules.
- Promote B1 to development evaluation only if the smoke is operationally valid; smoke success alone is not evidence of accuracy gain.

## Next Concrete Action

Implement `b1-structured-json-v1`, run local tests, then execute the bounded answer-blind V100 smoke.
