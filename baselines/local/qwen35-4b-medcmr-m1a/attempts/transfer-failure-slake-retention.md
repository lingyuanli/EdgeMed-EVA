# Transfer-failure Slice A: SLAKE retention

Status: `completed_positive / not a Med-CMR score`  
Date: 2026-09-02  
Compared systems: frozen B0 versus frozen M1a seed `20260903` / 128 steps

## Contract

- Dataset: admitted SLAKE English validation, 1,053 rows / 96 images, never used for M1a training.
- Prompt: answer-only external-retention variant, deterministic decoding, 32-token cap.
- Parser: accepts `Answer: value` or one unique nonempty line of at most 20 whitespace tokens only for this variant. Direct Med-CMR open parsing is unchanged.
- Metrics: normalized exact and token F1 proxies. These are not the unavailable official Med-CMR open judge metrics.
- Gate: both paired CI lower bounds must be at least `-1.0` point for non-inferiority.

## Result

| Metric | B0 | M1a | Delta | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| normalized exact | 46.0589% | 54.7009% | +8.6420 | [6.6477, 10.7312] |
| mean token F1 | 53.7949% | 59.3793% | +5.5843 | [3.8774, 7.3331] |

Predeclared answer-type slices:

| Slice | Count | B0 exact | M1a exact | Delta |
|---|---:|---:|---:|---:|
| CLOSED | 422 | 78.9100% | 82.2275% | +3.3175 |
| OPEN | 631 | 24.0887% | 36.2916% | +12.2029 |

Both runs completed 1,053/1,053 unique samples with `run_manifest.status=completed`, `metrics.complete=true`, and exit code zero. B0 emitted 546 marked plus 507 bare answers; M1a emitted 34 plus 1,019. There were no invalid parses under the frozen variant-scoped parser.

## Interpretation

The non-inferiority gate passed by a wide margin. M1a does not show general open-answer semantic forgetting on this human-annotated cross-source set. This weakens H3 and weakens a broad version of H1, but it does not explain the Med-CMR regression or prove commercial-model superiority. Option/schema sensitivity remains a live, externally testable explanation and is evaluated next.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| B0 run manifest | `729f39a9b3d574d6528d3eb965062215abb714fb865bcba0ab6fc8a5f34771b3` |
| B0 predictions | `01b5537f9bb0617654f10ccc8982f86c75c6ab7d4283c0dde20bd61a9ffb34c0` |
| B0 metrics | `7526125ef33f3b539196039b6b842c7c23135b4004dcf8d3ffd127e93892fb40` |
| M1a run manifest | `b31136f78848e50500151bac6530f381f0041657314b823d9a8cc194be9cd43e` |
| M1a predictions | `92bab3bd8e3eb8c9a877132fc6fed298c8a2f860a859842217c2cb3a0261c4f6` |
| M1a metrics | `1407d86325354d6db7a183269666968c07bc2b3f1d368a2fc7708b1a1b6b7e0b` |
| paired comparison | `85352ee9b5cc5eff509a12d2ea0dcb35e45881c8c1b09c54f61e3a4ee624163c` |

