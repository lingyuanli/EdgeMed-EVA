# Survey: transfer failure, choice bias, and source-diverse medical VQA

Survey date: 2026-09-02. This note records source findings used to design the M1a failure-analysis campaign; it does not treat paper claims as local experimental results.

## Findings

1. **LVLM multiple-choice selection bias.** The arXiv paper *Calibrating Selection Bias in Multiple-Choice Questions of Large Vision-Language Models* reports token- and position-related selection bias and sensitivity to option ordering in LVLM multiple-choice evaluation. This supports an answer-preserving option-rotation diagnostic for the A-D letter-trained adapter. Source: <https://arxiv.org/abs/2509.16805>.

2. **VQA under joint visual/text distribution shift.** *VQA-GEN: A Benchmark for Generalizing VQA Models* explicitly studies distribution shifts in both visual and textual inputs and motivates held-out-domain evaluation rather than a same-source validation-only gate. Source: <https://arxiv.org/abs/2311.00807>.

3. **Five-option external candidate.** The official `TsinghuaC3I/MedXpertQA` dataset card describes a multimodal subset with 2,010 image-text questions, five choices, clinical reasoning, and MIT terms. The visible split/label availability must be audited before it can be used as a development source; it is not automatically a legal public-test tuning set. Source: <https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA>.

4. **Held candidate.** The currently visible `neuripsqgr2026/Med4VQA` card claims grounded reasoning with boxes/masks across four modalities, but it is anonymous and its narrative count (137,960) conflicts with the viewer count (13,796). It remains survey-only until source identity, revision, license, and counts are independently resolved. Source: <https://huggingface.co/datasets/neuripsqgr2026/Med4VQA>.

## Local design consequence

- Treat PMC-VQA dev as a same-generation-family checkpoint gate, not a transfer gate.
- Add SLAKE human-annotated cross-source retention before more training.
- Add deterministic option rotation before interpreting MCQ gains as reasoning gains.
- Do not claim that a 4B model surpasses commercial models until a frozen, source-diverse development protocol and one-shot official evaluation both support it.

