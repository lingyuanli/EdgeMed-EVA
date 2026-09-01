# Med-CMR Test Firewall

## Purpose

The released Med-CMR MCQ set is the frozen leaderboard test surface. Its questions, images, references, per-sample correctness, and evaluator feedback must not become prompt-development, training, preference-generation, retrieval, or checkpoint-selection inputs.

## Allowed Before Candidate Freeze

- manifest-only input/schema checks without reference files;
- operational smoke that reports completion, unique IDs, schema validity, parse status, latency, memory, and hashes;
- aggregate facts already frozen by B0, such as overall/task accuracy and aggregate parse counts;
- development on separately sourced data after license, provenance, patient/article/image/text overlap, and near-duplicate checks pass.

## Prohibited Before Candidate Freeze

- reading Med-CMR reference answers while changing a prompt or method;
- selecting examples because B0 got them right or wrong;
- creating training or preference pairs from Med-CMR test outcomes;
- using per-sample test correctness to tune parsers, thresholds, crops, prompts, hyperparameters, or checkpoints;
- repeated full-test evaluation to choose among candidates.

## Role Separation

- the inference runner receives only the answer-free manifest;
- operational smoke must not invoke the scorer or mount the references path;
- the development operator uses only the external development manifest and its references;
- the release evaluator may access Med-CMR references only after candidate commit, prompt hash, model/checkpoint hash, run contract, and evaluation budget are frozen;
- release outputs expose aggregate metrics to development; per-sample test labels remain sealed.

## Evaluation Budget

One full Med-CMR evaluation is allowed per preregistered milestone: frozen B1 diagnostic, frozen best SFT checkpoint, frozen best preference checkpoint, and frozen final Agent. A failed infrastructure run does not consume the budget if hashes prove no scientific result was exposed. A completed test result consumes the budget even if it is poor.

## Track Boundary

Any method trained or selected using Med-CMR samples, references, or per-sample evaluator feedback moves to the separately named `in_domain_research` track. It cannot be compared as leaderboard-clean against paper-reported models.
