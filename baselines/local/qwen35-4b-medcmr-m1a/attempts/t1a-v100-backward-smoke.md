# T1a Answer-only QLoRA V100 Backward Smoke

Date: 2026-09-01  
Hardware: 1x Tesla V100-SXM2-32GB  
Final verdict: **backward/save smoke passed; model quality not yet evaluated**

## Frozen successful run

- Run: `qwen35-4b-pmc-t1a-qlora-smoke-v3-20260901`.
- Code commit: `12030339263ad368c90e5b467b001dfbf521f28d`.
- Contract SHA-256: `fcf450a961f75eb3314b24f3e0afac3bd1e2cb6f091a892fde514f793b7733d0`.
- Run-manifest SHA-256: `0c32fc0781c6d02ef54b4b915cf53b19266b0bdf5142b1eabdb589b538acfd07`.
- Training surface SHA-256: `2540d8527be99908967add8e4540cb8edc48d1d61c20d240303378587a6d847e`.
- References SHA-256: `cc1b985b5715ef45309bfe274dc0e6b48f6a9a0710b7a95bb4cfded62b319246`.
- Objective: assistant answer tokens only; direct prompt; 4 admitted examples;
  2 optimizer steps; gradient accumulation 2.
- Base: NF4 double quantization with FP16 compute; LoRA rank 16/alpha 32;
  language layers only; vision encoder and projector frozen.
- Uniform image cap: 786,432 pixels, aspect-preserving Lanczos.
- V100 FP16 GradScaler initial scale: 1.0.

## Observed evidence

| Step | Last-example loss | Gradient norm | Scale before/after | Applied |
|---:|---:|---:|---:|---|
| 1 | 0.7994397 | 42.5208626 | 1.0 / 1.0 | yes |
| 2 | 0.0096204 | 3.0455315 | 1.0 / 1.0 | yes |

- All four example losses were finite; first 0.9135091, last 0.0096204,
  mean 0.5561219.
- Trainable parameters: 32,464,896 of 2,622,558,720 parameters exposed by
  the quantized PEFT model.
- Peak allocated CUDA memory: 6,775.23 MiB.
- Training-loop elapsed time after construction: 7.64 seconds.
- Adapter weights SHA-256:
  `e319e00ff9658d967c5196d4df8829f43307b4d460cabe8e08655b81b89de08f`.

## Failure and falsifiable repair

The first run incorrectly exited zero despite `grad_norm=NaN` on both steps.
The default GradScaler initial scale was 65,536. A hardened v2 run with scale
128 failed at the first non-finite-gradient gate. Holding samples, model,
objective, and optimizer fixed while changing only the scale to 1 produced two
finite, applied steps; the final v3 reproduced the same adapter hash under the
final code commit. The confirmed failure cause is FP16 gradient overflow at the
higher initial scales on this V100 path, not GPU capacity.

The implementation now treats non-finite gradient norms or a decreased loss
scale as a hard failure. Failed runs and logs remain preserved and are not
promoted.

## Boundary and next gate

This proves construction, backward, optimizer update, finite gradients, memory
fit, and adapter serialization on one 32GB V100. It does not prove generalizing
accuracy or full-epoch stability. Next, reload this adapter for a bounded
inference smoke, then run a versioned pilot and compare it against the frozen
direct PMC-VQA development predictions before any Med-CMR test use.

The subsequent reload gate completed 4/4 deterministic direct-prompt
predictions with zero invalid parses and a completed hash-bound run manifest.
Its 3/4 accuracy is not used as an efficacy result because the sample is only
an operational slice. Adapter load compatibility is therefore passed; the next
efficacy gate is the frozen 128-step pilot on all 512 development examples.
