# External Training And Development Data Decision

## Decision

Freeze two complementary sources for the first optimization cycle:

1. **PMC-VQA v2 seed pool**: caption/report-derived multiple-choice supervision
   at scale. It can teach compact output contracts and broad visual question
   coverage, but it is not a human-evidence gold set.
2. **PMC-VQA v2 test development cohort**: 512 deterministic, article-disjoint
   MCQs for comparing B0 direct output and B1 structured output.
3. **SLAKE English validation**: an independently versioned, human-annotated
   cross-dataset check. It is never mixed into training.

MS-CXR and VinDr-CXR remain later candidates for spatial evidence evaluation.
Their access controls make them unsuitable as hidden dependencies of the first
reproducible run.

## Fit to the research claim

| Requirement | PMC-VQA v2 | SLAKE validation | Consequence |
|---|---|---|---|
| Multimodal medical QA | yes | yes | both exercise image-conditioned answers |
| Scale for low-cost SFT | high | modest | train on PMC-VQA; cross-check on SLAKE |
| Human gold evidence | no | semantic labels, no universal boxes | do not claim evidence localization |
| Immutable public revision | yes | yes | source files can be hash-bound |
| License usable for cycle 1 | article join plus CC BY-SA output | CC BY 4.0 | admission is fail-closed |
| Med-CMR answer independence | must be checked | must be checked | run exact/near overlap gate before use |

## Frozen cohort rules

### PMC-VQA seed pool

- Read only `train_2.csv` at revision
  `b56ae594f794867893143b337b4118a835794647`.
- Accept official train rows whose answer is A-D, whose choices and image are
  present, and whose PMCID has an allowed license in the pinned PMC file.
- Preserve audit fields. Label supervision `synthetic` and evidence source
  `caption-derived`.
- Derive article group hashes from PMCID. This is a source-group boundary, not
  a claim that the paper exposes patient identifiers.
- Rank records by SHA-256 of a frozen seed plus source row ID. Med-CMR scores,
  labels, and outputs never enter selection.

### SLAKE gold development pool

- Read only `validation.json` at revision
  `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`.
- Retain English records with a present image and non-empty question/answer.
- Keep the official validation split intact and use image name as group boundary.
- Never train on these records or select candidates using a Med-CMR test score.

### PMC-VQA MCQ development cohort

- Use only `test_2.csv`; frozen inspection found zero exact image-name and zero
  PMCID overlap with `train_2.csv`.
- Select 512 records with seed `edgemed-pmc-vqa-v2-dev-20260901`, at most one
  question per image, and exclude normalized exact questions present in the
  frozen 2,000-record train seed. This is the primary B0-versus-B1
  prompt-selection surface.
- Call it synthetic MCQ development, not human gold. SLAKE remains the human
  cross-dataset check.

## Admission sequence

1. Download named source files at frozen revisions.
2. Verify byte count and SHA-256 before extraction or parsing.
3. Build separate train-seed, MCQ-development, and SLAKE manifests with
   per-image SHA-256 and source metadata.
4. Run the overlap validator against the answer-free Med-CMR manifest.
5. Quarantine every missing-license, missing-file, or overlap finding.
6. Freeze the accepted manifest hash before candidate training.

Passing this sequence establishes provenance and test independence. It does not
establish annotation correctness, clinical safety, or baseline superiority.

Exact URLs, revisions, sizes, hashes, and observed schema are recorded in
[`sources/research_external_medvqa_datasets.md`](../../../sources/research_external_medvqa_datasets.md).
