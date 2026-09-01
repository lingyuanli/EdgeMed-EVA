# SLAKE English Validation Admission Receipt

Run date: 2026-09-01  
Execution host: `ubuntu@117.50.188.27`  
External data root: `/home/ubuntu/data/external/slake-a9083ce6`

## Bound source

- dataset: `BoKelvin/SLAKE`
- revision: `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`
- `validation.json`: 639,139 bytes,
  SHA-256 `32b016440b0c3be11056a78a18eeab46333268407fbb6e6b32f9f4c2debc50f6`
- `imgs.zip`: 212,343,373 bytes,
  SHA-256 `44eb7d9214e1ac5b7946e237b669401866df9114fd9a3618c5a84fbffcded0b0`
- extraction: 2,592 files / 268,343,111 uncompressed bytes
- extraction report SHA-256:
  `b44bc393f6adbb06255902957a929fcebb7d95f26cbc95f5414bd4af19131454`

## Manifest

- accepted English questions: 1,053
- rejected non-English questions: 1,046
- unique referenced images: 96
- manifest SHA-256:
  `f6ab8734e7df82a83ca59a7b5c93a03fc027bb4dc8fe8f369f34401bdbcfa7a0`
- build report SHA-256:
  `661ff18e0f24bbeb9e545e49d2c334ac5c747d7e707f340153a24d8553c00308`

## Med-CMR overlap gate

The gate used only the 16,655-row answer-free Med-CMR MCQ manifest, SHA-256
`9ec6f833f1f53509d25873b2beb77960f18d55b4b514a0f4796efd147d0219d7`.
No Med-CMR reference or score was loaded.

- file/integrity problems: 0
- exact or near-text confirmed overlaps: 0
- exact-image confirmed overlaps: 0
- dHash≤4 candidates: 278 record-level findings
- unique candidate image pairs: 24 across 9 SLAKE images
- second-stage threshold: contrast-normalized pixel correlation ≥0.98
- maximum observed candidate correlation: 0.8451463418980412
- confirmed near-image overlaps: 0
- candidate-pair visual audit: all 24 are distinct acquisitions; most are
  chest radiographs with similar global anatomy, plus one nonmatching brain pair
- final gate: `passed`
- final gate report SHA-256:
  `88d52c18967bff1c6e8fe313eb85d0d6ea9544faf4a3ff053cfaaf4109e2f619`

The first one-stage dHash report failed conservatively and is retained at
SHA-256 `9bff90cda2902e4e3ed5081896e5d9e2c098b1ff45d1bc1d22436b04ed3e92e7`.
It was not treated as a dataset leak or as a passing result. The repair added a
pre-score confirmation criterion and a synthetic hash-collision regression test.

## Scope of promotion

SLAKE English validation is admitted only as an external, cross-dataset check.
It has no MCQ choices and therefore cannot select between the direct and B1
structured MCQ prompts. That primary comparison remains blocked on the disjoint
PMC-VQA v2 test cohort.

