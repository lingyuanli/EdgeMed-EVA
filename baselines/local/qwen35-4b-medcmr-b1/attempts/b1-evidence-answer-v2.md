# B1 Evidence + Answer v2 Smoke Receipt

## Verdict

- status: `smoke_passed`
- scientific score: not computed
- test firewall: `PASS`; no references path and no scorer were used
- promotion: allow development-set evaluation after the external data provenance/overlap gate passes
- full Med-CMR evaluation: not authorized by smoke success

## Frozen Run

- host: `ubuntu@117.50.188.27`
- code commit: `c926fbb75df72e21a7a3ba9bb7b8777db1ef4845`
- run directory: `/home/ubuntu/EdgeMed-EVA/runs/qwen35-4b-medcmr-b1-evidence-answer-v2-smoke-20260901T0630Z`
- prompt variant: `evidence_answer_v2`
- prompt SHA-256: `65054cc9d881fd95adaf963fad56783b299d9d934696d3abefd18805a14697d1`
- contract SHA-256: `0cbebcdaeeb8be6c29509ef66ced83515f1b8a61ee0f32f1b34bb859009cb311`
- run manifest SHA-256: `43cd8123b7dee4d1762a80b3bc5d1622c75a4de4eea9c58a0c2407c5f0ba2169`
- events SHA-256: `6aea7430f7c89c08ea31988123ddfeda2a1a0745ae0c887389c8583d3386de6b`
- predictions SHA-256: `f284145ace932a92b7cc692edc59b7e7e99e54d62003f3fcca2b51773c8e1ed4`
- process exit code: `0`

## Operational Result

- completed: `14/14`, 14 unique IDs, all predictions contract-bound
- strict evidence/answer JSON: `14/14`
- non-empty observations and non-null answers: `14/14`
- observations at most 20 whitespace-delimited words: `14/14` (range 7–16)
- inference time: `31.3929 s`; mean per-sample latency `2.2404 s`
- mean output tokens: `27.2857`
- peak allocated GPU memory: `3,420.08 MiB`

Compared with the archived v1 operational smoke, v2 changes only the structured output surface: it removes the unreliable zero-shot competing-hypothesis list and requires one short observation plus one answer. Strict parse rate rose from 8/14 to 14/14, mean output length fell from 72.14 to 27.29 tokens, and mean latency fell from 5.42 to 2.24 seconds.

This comparison is format-only. No reference answers were loaded, so it cannot support an accuracy or medical-evidence claim. The next valid measurement surface is an external, overlap-audited development set.
