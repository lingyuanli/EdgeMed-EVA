# B1 Structured JSON v1 Smoke Receipt

## Verdict

- status: `smoke_failed`
- scientific score: not computed
- test firewall: `PASS`; the run command had no references path and did not invoke the scorer
- route: one bounded format repair is justified; do not rerun v1 unchanged

## Frozen Run

- host: `ubuntu@117.50.188.27`
- code commit: `7422cf0d1b0ab8a0cf7ff4f5b8014239fa10f03b`
- run directory: `/home/ubuntu/EdgeMed-EVA/runs/qwen35-4b-medcmr-b1-structured-smoke-20260901T0615Z`
- prompt variant: `structured_evidence`
- prompt SHA-256: `924e4a861273f1c76e8211e2dccfdb65ed54d8c3a185ea2956544e42c0e5353f`
- contract SHA-256: `0a95eb658e89e58f68bc81561b7bf2dca672d140926c451039991b7bf78525bb`
- run manifest SHA-256: `a3c95627447a377327fc6631618e24b4e83ee5e4d8f92d2afc81a2fec4ca9c82`
- events SHA-256: `544551d751b5f4e7eed65fc7dc13f892341b664106ec38b0311abe16bbdb8d72`
- predictions SHA-256: `5ede47c66a337bb97d5ad78afcc7c22b505d940f17dcde7479f0700964382704`
- process exit code: `0`

## Operational Result

- completed: `14/14`, 14 unique sample IDs
- inference time: `75.8661 s`; mean per-sample latency `5.4171 s`
- mean output tokens: `72.1429`
- peak allocated GPU memory: `3,420.09 MiB`
- strict structured JSON: `8/14`
- invalid structured schema: `4/14`
- invalid/truncated JSON: `2/14`

No reference answers or per-sample correctness were read. Raw-output inspection was limited to format diagnosis. Four otherwise valid JSON objects contained 4–5 hypotheses, exceeding the prompt/schema limit of 1–3. Two responses spent the 128-token budget on verbose observations and ended before a complete JSON object was produced.

## Root Cause And One Repair

The untrained 4B model does not reliably satisfy a three-field schema that combines free-text evidence with a bounded competing-hypothesis list. The smallest defensible repair is to reduce B1 to the essential attribution surface—one short observation plus one answer—while preserving the model, data, preprocessing, deterministic decoding, answer isolation, and strict no-repair JSON parser. Competing hypotheses remain a training-stage target rather than being claimed as a zero-shot capability.

Archive v1 after this receipt. Do not relax the parser to accept arbitrary candidate counts and do not guess answers from truncated JSON.
