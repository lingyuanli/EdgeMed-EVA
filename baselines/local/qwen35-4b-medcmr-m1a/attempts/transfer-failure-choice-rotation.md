# Transfer-failure Slice B: PMC choice rotation

Status: `completed / H2 supported`  
Date: 2026-09-02  
Transform: deterministic cyclic shift `A->B->C->D->A`, preserving option content and remapping references

## Result

| Metric | B0 | M1a | M1a - B0 | Paired 95% CI |
|---|---:|---:|---:|---:|
| rotated accuracy | 55.4688% | 57.4219% | +1.9531 | [-2.1484, 6.0547] |
| content consistency, original to rotated | 71.8750% | 63.2813% | -8.5938 | [-13.2813, -4.0967] |

The paired content-consistency comparison has exact McNemar `p=0.0004485`. Both runs completed 512/512 unique samples with exit code zero. B0 had 3 invalid rotated parses; M1a had zero.

Original frozen PMC accuracy was B0 `57.6172%` (295/512) and M1a `66.0156%` (338/512). After an answer-preserving rotation, B0 lost `2.1484` points while M1a lost `8.5938` points. M1a's original advantage shrank from `8.3984` to an inconclusive `1.9531` points.

## Interpretation and decision

H2 is supported: answer-letter SFT materially amplifies option-order/label sensitivity. The experiment does not claim this is the only cause of the Med-CMR regression, but it is a falsified safety property of the current objective. No more single-letter answer SFT is allowed. M1a may remain a domain-adaptation result, but it is not a safe parent for a benchmark claim.

The next training candidate must supervise option content or semantic answer text and must pass both original-order and rotated-order development gates. SLAKE retention remains a co-primary safety gate. No Med-CMR run is authorized.

## Scorer repair audit

The first paired consistency file incorrectly counted three B0 `None -> None` invalid pairs as consistent. It was preserved as `*.buggy-v1.json` with SHA-256 `c1d82507…42d82`. Commit `3f95b2a` added a regression test and required a mapped non-null expected answer. Only the paired statistic was recomputed; GPU predictions were unchanged. The corrected file SHA-256 is `18d1aa36…c1861`.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| rotation report | `6c5e78aa8ef7bd94895fe40facbf8305808cf66d39252bd6ff377eea62f693dc` |
| B0 run manifest | `bf9f089fb51164055a8afcd9a5d22d954616713877d34d2d21be7cb93dcabe03` |
| B0 predictions | `b4cefd19bc7a36cebf86775e7e0b98885218d32f7dbc99d835d18d1aff07ba4b` |
| B0 metrics | `58b91999f17d57487631a96df35cb3437c95c2e841bf6a99ff6a4436de97c2fd` |
| B0 invariance | `dc841db8eb35ea8353b0d9df747c9d754743838c184f73cf2f83db006ca3cff9` |
| M1a run manifest | `a25b8447fea809c7fba9e3c993ad74a27f119f55af47173019cbd65895fef7b4` |
| M1a predictions | `91d0c27604f90179b4020c63112ff06680b4ff0fc82c846e9f45e2b0a8ac8b9b` |
| M1a metrics | `959045fe2bf3580bafe21e10bb8b71b8b3e27e30aab168719abc711c566a9507` |
| M1a invariance | `3d78f72269f2d7d467583aed14f66b446f781754af4039d940bcfb1bd75a830a` |
| rotated accuracy pair | `b44c2af282193036d72efc38578f168f97006aef536966c3e96b8ae939cda2c2` |
| corrected invariance pair | `18d1aa3674d9aed6c125d3ac4f12ef0f5dc0bd8db0ae02914a6be05cf4fc1861` |

