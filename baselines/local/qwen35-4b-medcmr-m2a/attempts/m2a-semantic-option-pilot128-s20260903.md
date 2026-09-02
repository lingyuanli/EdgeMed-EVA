# M2a Semantic Option-Content Pilot, seed 20260903

Status: `completed / accuracy gates passed / invariance gate inconclusive`

Date: 2026-09-02

Parent weights: frozen Qwen3.5-4B B0, not M1a

Single changed variable: assistant target changed from one answer letter to the complete visible option text

## Result

All four evaluation runs completed 512/512 unique PMC development samples. Corrected invalid counts are B0 `12/8` and M2a `10/7` on original/rotate-1. These rows were already scored as incorrect, so accuracies and paired comparisons are unchanged.

| Surface | B0 | M2a | M2a - B0 | Paired 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| original order | 58.0078% | 62.8906% | +4.8828 | [0.7813, 8.9893] | 0.0261 |
| rotate-1 order | 55.6641% | 60.1563% | +4.4922 | [0.5859, 8.3984] | 0.0346 |

Content consistency from original to rotate-1 was 70.5078% for B0 and 72.2656% for M2a. The paired M2a-minus-B0 delta was +1.7578 points, but its 95% interval `[-3.1250, 6.6406]` crossed the preregistered -1 point non-inferiority margin (`p=0.531`).

## Gate decision

- `PASS`: original-order candidate gain is positive.
- `PASS`: rotated-order paired interval lower bound is at least -1 point.
- `INCONCLUSIVE`: consistency interval lower bound is below -1 point.
- overall frozen gate: `failed` because all three checks were conjunctive.

M2a repairs the main M1a symptom: the improvement remains statistically positive after answer-preserving rotation, whereas M1a's advantage collapsed under rotation. The point estimate also improves consistency. However, this single 512-row comparison does not establish the preregistered consistency non-inferiority claim. M2a is therefore not promoted to SLAKE or Med-CMR.

The next candidate is M2b: retain the same semantic target and add only deterministic, training-only option-order augmentation. B0 evaluation outputs remain frozen and reusable. No per-sample correctness inspection is authorized.

## Training receipt

- optimizer steps: 128/128 applied
- examples seen: 256
- loss: finite; mean `0.183526`, last `0.147254`
- peak CUDA allocation: `8475.42 MiB`
- adapter model SHA-256: `d41ac00e2357099955a539f4980a698da2f36c1515338023014b5d278681c6c0`
- run manifest SHA-256: `cd7e0c69ac1fd381d5b3de7b913f3e162238b2e556a0b1769ea9529e7668086e`

The first parser-v1 evaluation was stopped after 66 B0 rows and preserved; its predictions SHA-256 is `bdaf1a0cc01907556256275f91702d9a00f8e905e3925d6d085c7edf65f68c20`. Parser v2 accepts a labelled answer only when its copied content agrees with that visible option. The adapter was reused without retraining.

## Evaluation artifact hashes

| Artifact | SHA-256 |
|---|---|
| B0 original corrected metrics | `be55b6a78de7743186b0160d58bb2fb9d3348770c15ec0260b7e7b41e5721f9a` |
| B0 rotate-1 corrected metrics | `636eaa28aed5a9849b2e7a25453773c68b38d966cb0729d8dc9cf111bdb200eb` |
| M2a original corrected metrics | `40e10acbbc270185ccfea2179771174baeb5896c11e758a4eb7f48f66e59a777` |
| M2a rotate-1 corrected metrics | `02f157592a85fa148524dc7d1aff3db142b82a1b37e64fbb931ca1945948ae2d` |
| B0 consistency | `9dd527128c76fd0052b6b65b9884def91ed5f58a932d6c290eddc6e7c86aa0f9` |
| M2a consistency | `33bb3cc78fe1ca74d321f6c8a3b8036adbd28bc1074957eb23cb19445c53fd67` |
| original paired comparison | `b22321a3dee34f073b54f080dfebef339b9a51e289a455712ad920ccd2fbab50` |
| rotate-1 paired comparison | `ae0b23fd4e6b2bd2581dd66dfdec30c1a2cbf6189f9785c979e768398a9c30a0` |
| consistency paired comparison | `aa478c330ecabebd2bf21ab05178178964c4a10e649f6b31e2d59e041f947a36` |
| frozen gate receipt | `f6ad08e3bc10fc18086ffee712ea599ac730cc09fbdbc7f2196e21bf0121a71e` |

The pre-fix metric files are preserved with hashes `126c56db…0a88`, `809903b3…573`, `7c3f4d49…9bb2`, and `b33d1061…a28999`. Commit `a5ddf38` corrected only variant-specific invalid aggregation; no predictions, accuracy, or paired result changed.
