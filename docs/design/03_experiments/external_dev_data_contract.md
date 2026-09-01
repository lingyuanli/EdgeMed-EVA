# External Development Data Contract

## Run Contract

- tier: `auxiliary/dev`
- research question: can an external development record be admitted without provenance, file-integrity, or Med-CMR overlap ambiguity?
- null hypothesis: the candidate record cannot be proven independent of the frozen benchmark.
- alternative hypothesis: every accepted record has complete provenance, a verified image hash, and no declared, exact, or configured near overlap.
- baseline reference: frozen Med-CMR answer-free manifest; references are prohibited.
- primary gate: `report.status == passed`.
- stop condition: any schema, provenance, file, declared-overlap, exact-overlap, or near-overlap finding.
- resource budget: CPU only; no model inference or GPU allocation.

## Manifest Schema

Every external record must contain:

```json
{
  "record_id": "external-000001",
  "source_dataset": "dataset-name",
  "source_version": "immutable-version",
  "license": "license-id-or-receipt",
  "patient_group_hash": "non-reversible-group-hash",
  "image_path": "relative/path.png",
  "image_sha256": "...",
  "question": "...",
  "answer": "...",
  "annotation_type": "human|report-derived|synthetic",
  "quality_status": "accepted|quarantined",
  "benchmark_overlap": "none|suspected|blocked"
}
```

Only `accepted` records enter the overlap gate. `quarantined` records remain visible in the manifest but are excluded from development and training.

## Gate Checks

The validator fails closed on:

- missing/empty provenance, duplicate record IDs, invalid enums, or empty question/answer;
- missing images or declared-versus-actual SHA-256 mismatch;
- `suspected` or `blocked` overlap on an accepted record;
- exact image SHA-256 collision with Med-CMR;
- normalized exact question collision;
- near-text collision using a configured SequenceMatcher threshold after distinctive-token candidate retrieval;
- near-image collision using 64-bit difference hash for candidate retrieval,
  confirmed by contrast-normalized 128×128 pixel correlation. The frozen cycle-1
  thresholds are dHash Hamming distance ≤4 and correlation ≥0.98.

Near-duplicate heuristics reduce risk but do not prove semantic independence.
Candidate pairs rejected by the correlation confirmation remain in the report
for audit. Before admission, source/article/patient identifiers and a manual
audit of all candidate clusters remain required. Thresholds are frozen before
inspecting downstream model scores.

For large training pools, the conservative alternative to manual audit is to
mark every record in an unreviewed candidate cluster as
`quality_status=quarantined` and `benchmark_overlap=suspected`. The record stays
in the auditable manifest but is skipped by surface generation and training.

## Command

```bash
PYTHONPATH=src python -m edgemed_bench.validate_external_data \
  --manifest data/external/dev.jsonl \
  --data-root data/external/raw \
  --benchmark-manifest /home/ubuntu/data/medcmr/release_a9b2d6e6/manifests/mcq.jsonl \
  --benchmark-data-root /home/ubuntu/data/medcmr/release_a9b2d6e6/raw \
  --output artifacts/data-gates/external-dev/report.json
```

The report contains only IDs and overlap diagnostics, never Med-CMR answers.
