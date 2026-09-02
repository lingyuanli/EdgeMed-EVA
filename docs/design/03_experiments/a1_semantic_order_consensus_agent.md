# A1 Semantic Option-Order Consensus Agent

Status: `preregistered / implementation preflight`

Parent weights: frozen M2a seed-20260903 adapter

## Agent mechanism

The controller presents the same image and question under all four cyclic option orders. Each worker call uses the frozen `semantic_option` prompt and parser. The controller maps every visible answer back to the canonical semantic option identity, counts votes, and emits one answer plus an answer-free trace containing per-view parse status, canonical vote counts, consensus margin, and tie state.

Ties are resolved by SHA-256 of normalized option content, never by option letter, view order, reference answer, or model score. All-invalid cases remain invalid. The aggregation is therefore invariant to the ordering of its four input views. This is an Agent/controller change only: no model training, prompt change, reference access during inference, or per-sample correctness routing.

## Frozen inputs and compute

- M2a adapter: seed 20260903 / 128 steps; adapter SHA-256 `d41ac00e2357099955a539f4980a698da2f36c1515338023014b5d278681c6c0`.
- existing views reused: PMC M2a original/rotate-1 prediction SHA-256 values `bb94e5a80084cc16a6a0fae730fa68f9f59067743e1998154edad2fb62ce7648` and `3bd87bee4df3ded0da09d540068776fc4163a8fab3626748fd79c93016a62880`.
- new inference: only rotate-2 and rotate-3 PMC surfaces, 512 rows each.
- total steady-state cost: four model calls per question; incremental pilot cost is two calls because original and rotate-1 are frozen and reused.
- model, image preprocessing, decoding, 64-token cap, option-text parser, dev references, and scorer remain unchanged.

## Preregistered gates

1. Unit gate: content preservation for shifts 2/3, exact sample-ID equality, reference-free controller, correct inverse mapping, view-order invariance, content-hash tie break, and no guess on all-invalid.
2. Eight-row operational smoke for both missing views; 8/8 completion each and at least 7/8 parseable each. Correctness is not inspected.
3. One full 512-row run for rotate-2 and rotate-3 only; existing original and rotate-1 predictions must match their frozen hashes.
4. Agent output gate, conjunctive:
   - A1-minus-B0 original-order paired 95% CI lower bound is above zero;
   - A1-minus-M2a single-view paired 95% CI lower bound is at least -1 point;
   - all-invalid rate is no greater than M2a original's corrected invalid rate;
   - reversed view-argument aggregation produces identical `(sample_id, parsed_answer)` pairs.
5. If the PMC gate passes, admit a separate source-diverse MCQ retention set before any Med-CMR milestone. SLAKE is open-answer and cannot validate this MCQ-only controller.

Failure archives A1. The four-view cost must be reported alongside accuracy; a successful accuracy result is not an edge-efficiency result until a two-view or early-stop policy is evaluated under a separately frozen compute/quality frontier.
