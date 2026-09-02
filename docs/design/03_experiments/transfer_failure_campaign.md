# M1a Transfer-Failure Analysis Campaign

Campaign ID: `transfer-failure-20260902`  
Parent: frozen `m1a-answer-qlora` seed `20260903`, 128 steps  
Campaign status: `completed / H2 supported / semantic-objective redesign next`

## 1. Boundary and question

M1a is not a failed optimizer run. It produced finite gradients, applied steps, a reloadable hash-bound adapter, and positive paired gains on three PMC-VQA development seeds. It nevertheless regressed on the one allowed Med-CMR milestone evaluation: `24.3591%` versus B0 `27.1690%`, delta `-2.8100` points with a paired interval entirely below zero. All seven Med-CMR dimensions regressed while invalid parsing fell from 773 to zero.

The campaign asks one narrower question: **which externally testable failure mode made a same-source PMC gain non-transferable?** Med-CMR per-sample correctness is sealed and is not an analysis surface.

## 2. Frozen hypotheses

| ID | Falsifiable explanation | Discriminating external slice | What would weaken it |
|---|---|---|---|
| H1 | The adapter learned the synthetic, caption-derived PMC generation distribution rather than transferable medical visual reasoning. | Human-annotated SLAKE English validation, never used in training, split by OPEN/CLOSED and content type. | M1a is non-inferior to B0 on both normalized exact and token F1, including OPEN questions. |
| H2 | Answer-only A-D supervision created option-letter or position shortcuts that do not survive schema/order changes. | Deterministic answer-preserving option rotation on the frozen PMC dev set, evaluated for content-consistency and accuracy under rotation. | M1a retains content-equivalent answers at least as well as B0 and its original PMC gain survives rotation. |
| H3 | The main observed benefit is format compliance, while semantic answer capability is retained poorly outside single-letter MCQ. | SLAKE open-answer retention plus parse-status accounting. | M1a improves or is non-inferior on semantic metrics without merely changing parse validity. |

These are candidate explanations, not conclusions. Aggregate Med-CMR results alone cannot identify one as causal.

## 3. Execution order and stop rules

### Slice A — cross-source open retention (claim-critical)

- Data: admitted SLAKE validation English, revision `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`, 1,053 rows / 96 images.
- Comparison: frozen B0 versus frozen M1a seed3/128 under the identical answer-only external-retention prompt, image budget, deterministic decoding, and conservative `Answer:` parser. This prompt is frozen before reference scoring and is not the Med-CMR official open prompt.
- Metrics: normalized exact, token F1, parse status, and predeclared `answer_type`, `base_type`, `content_type`, and `modality` slices.
- Metric label: external retention proxy only; it is not the unavailable official Med-CMR open judge.
- Operational gate: B0 and M1a must each produce at least 31/32 parseable short answers within 32 tokens on the same answer-blind smoke. `Answer: value` and a unique nonempty line of at most 20 whitespace tokens are accepted only for this answer-only variant; the direct Med-CMR open parser is unchanged. The earlier direct-reasoning prompt at 64 tokens was stopped after 51 rows because 46 were truncated before `Answer:`; a bounded 128-token retry reached only 29/32. The first answer-only smoke showed that B0/M1a usually emit valid bare single-line answers despite omitting the requested marker, so its parser contract failed at 15/32 and 1/32. All failed directories and hashes remain preserved; no references were inspected.
- Passed preflight: B0 `32/32` parseable (`15 answer_only + 17 bare_answer`, predictions SHA-256 `ad8b330d…1834f`); M1a `32/32` (`1 + 31`, `ec2bd9b2…e6f7a`). Full Slice A launched only after this receipt passed.
- Completed result: exact `46.0589% -> 54.7009%` (`+8.6420`, paired CI `[6.6477,10.7312]`); token F1 `53.7949% -> 59.3793%` (`+5.5843`, `[3.8774,7.3331]`). H3 is weakened and broad cross-source semantic forgetting is not supported. Slice B is now claim-critical.
- Promotion gate: the 95% paired interval lower bound must be at least `-1.0` point for both exact and token F1. Any interval wholly below `-1.0` archives M1a as a general-retention parent.

### Slice B — answer-label invariance (diagnostic)

- Data: existing overlap-audited PMC-VQA dev 512, never Med-CMR.
- Transform: one deterministic cyclic option rotation, with reference letters remapped to preserve answer content.
- Metrics: rotated accuracy, original-versus-rotated content consistency, and B0-versus-M1a paired difference.
- Stop rule: if M1a loses its original gain or its content consistency is materially below B0, do not add more answer-letter SFT; move to option-text/semantic objectives.
- Frozen implementation: `scripts/run-pmc-choice-rotation.sh`; the transform records the old-to-new label mapping without exposing the reference answer to inference.
- Completed result: M1a rotated accuracy advantage shrank to `+1.9531` points with CI crossing zero. M1a content consistency was `63.2813%` versus B0 `71.8750%`; paired delta `-8.5938`, CI `[-13.2813,-4.0967]`, McNemar `p=0.0004485`. H2 is supported.

### Slice C — source-diverse five-choice admission (future gate)

- Candidate: MedXpertQA-MM because it provides five-option medical multimodal questions under MIT terms.
- Required before use: immutable revision, released-label check, license/provenance audit, patient/image/question overlap gate, and a written rule separating development selection from any public-test claim.
- Med4-VQA remains held: the currently visible anonymous card and inconsistent displayed row counts are not mature enough for selection-critical use.

## 4. Decision map

1. If Slice A fails, archive M1a as the parent for evidence/Agent training. Return to B0 and redesign the objective around semantic answer text plus source mixing.
2. If Slice A passes but Slice B fails, retain M1a only as a domain adapter candidate and replace letter-token supervision with answer-text or option-content supervision. **Observed branch: selected.**
3. If both pass, the Med-CMR failure remains unresolved distribution shift; admit a source-diverse five-choice gate before any new training.
4. No branch authorizes another Med-CMR evaluation. A new test run requires a separate frozen mechanism milestone and one-shot budget.

## 5. Frozen runtime and evidence

- Hardware: 1 x V100-SXM2-32GB; no extra GPU required.
- B0 model and revision, M1a adapter hash, prompt, deterministic decoding, image budget, and parsers remain frozen.
- Inference manifests exclude answers; references remain separate mode-0600 surfaces.
- Each run must emit contract, manifest, events, predictions, metrics, and file hashes.
- Failed or negative slices are archived, not overwritten.

## 6. First command

```bash
cd /home/ubuntu/EdgeMed-EVA
tmux new-session -d -s slake-m1a-retention \
  'bash scripts/run-slake-m1a-retention.sh > runs/slake-m1a-retention-20260902.log 2>&1; echo $? > runs/slake-m1a-retention-20260902.exit'
```
