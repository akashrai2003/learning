# Module 22 — Automatic Mixed Precision and Numerical Stability

**Phase:** 4 — PyTorch model engineering and optimization  
**Time:** 9 focused hours (three 3-hour sessions)  
**Prerequisites:** tensor algebra, autograd, optimizers, IEEE formats, and a small training loop.

## Existing inference curriculum bridge

Revisit [04 — Numerical formats and error](../../inference-engineering-curriculum/modules/04-numerical-formats-and-error.md), [08 — Normalization, residuals, and stability](../../inference-engineering-curriculum/modules/08-normalization-residuals-and-stability.md), [25 — FP8 training and inference](../../inference-engineering-curriculum/modules/25-fp8-training-and-inference.md), and [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). This module turns format knowledge into a reliable PyTorch training and inference practice.

## Why this matters and outcomes

Mixed precision is not a switch that makes a model “half precision.” It is a dataflow policy: each operation gets an execution/accumulation type, parameter storage has another type, gradients have another, and optimizer state often remains FP32. You will learn to select and audit those policies rather than treating a speedup as proof of correctness.

By the end, you can explain overflow, underflow, cancellation, and loss scaling; use the current unified `torch.amp` APIs; find the first non-finite tensor; compare BF16 and FP16 deliberately; and prove numerical and throughput effects with a pinned experiment.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P22 — Automatic mixed precision and numerical stability**.

1. [AMP documentation](https://docs.pytorch.org/docs/stable/amp.html) — learn the API or system contract.
2. [Mixed Precision Training paper](https://arxiv.org/abs/1710.03740) — study the underlying design, paper, or specification.
3. [Numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [TorchAO source](https://github.com/pytorch/ao) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Torchtune source](https://github.com/pytorch/torchtune) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

For a linear layer, the useful contract is not simply \(Y=XW^T\). Inputs and weights may be cast to a low precision type, products are usually accumulated in a wider type, then the result is rounded on store:

\[
\hat y_{ij}=\operatorname{round}_{o}\left(\sum_k \operatorname{round}_{a}(x_{ik})\operatorname{round}_{b}(w_{jk})\right).
\]

FP16 has more mantissa precision near one but a smaller exponent range than BF16; BF16 shares FP32’s exponent width but has seven explicit fraction bits. Thus BF16 commonly avoids gradient overflow without loss scaling, while FP16 often needs it. Neither makes reductions, softmax, normalization, or optimizer updates automatically safe. A stable softmax uses \(m=\max_j z_j\), then \(\exp(z_j-m)/\sum_k\exp(z_k-m)\); implementing the naive form is an excellent intentional failure test.

Loss scaling replaces loss \(L\) with \(sL\), so backprop produces \(s\nabla L\). Before `optimizer.step`, divide gradients by \(s\). Dynamic scaling lowers \(s\) after an overflow and grows it after successful intervals. It does **not** repair a divergent model, an invalid target, NaN input, or a kernel bug. Gradient clipping must happen after unscaling.

Autocast selects eligible operations according to backend policy; it is not a promise that every tensor becomes low precision. Make the device type explicit, for example `torch.amp.autocast("cuda", dtype=torch.bfloat16)`, and use `torch.amp.GradScaler("cuda")` only where a scaler is appropriate. APIs and eligibility tables evolve, so pin `torch.__version__`, record `torch.__config__.show()`, and inspect the installed documentation/source rather than copying an old `torch.cuda.amp` recipe blindly.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/foundations/amp/train_fp32.py`: a tiny Transformer classifier trained FP32 with fixed seed, validation loss, gradient norms, and activation/gradient finite checks.
2. **Optimized:** add `train_amp.py` with a BF16 autocast path and a separate FP16-plus-GradScaler path. Keep optimizer, batch order, warmup, and evaluation precision fixed. Log per-step dtype, scaler value, maximum absolute activation/gradient, peak memory, and tokens/s.
3. **Integrated:** create `torch-lab/inference/amp_policy.py` for the decoder project. Exercise prefill and decode separately, assert output logits remain within an agreed tolerance, and profile whether the chosen GPU actually uses Tensor Cores for its matrix shapes.

## Three focused sessions

1. **Format laboratory:** implement naive and max-subtracted softmax; sweep logits and compare FP32, FP16, and BF16 error, overflow point, and `isfinite` results.
2. **Training policy:** train the same small model in FP32, BF16, and FP16/scaled mode. Inject an overly large loss once, locate the first non-finite value with hooks, and distinguish a skipped step from a successful update.
3. **Inference evidence:** profile a representative matmul/attention block with `torch.profiler`; explain tensor-core eligibility, memory savings, cold/warm timings, and accuracy drift.

## Experiments and deliverables

Commit `amp_matrix.json` with GPU, driver, PyTorch build, dtype policy, shape, seed, loss curves, validation metric, non-finite events, peak allocated/reserved memory, and warm throughput. Add `test_amp_parity.py` with tolerances justified by observed FP32 variation, not a guessed decimal count. Write a one-page incident note for one induced overflow.

## Common failure modes

- Comparing FP16 and BF16 while silently changing batch size or optimizer state.
- Calling `scaler.step` without unscaling before clipping.
- Declaring a model stable because loss is finite while logits or gradients are already saturated.
- Benchmarking first-call compile/autotune time as steady-state inference.
- Forcing an unsupported dtype or assuming consumer-GPU FP64/FP8 behavior mirrors a datacenter accelerator.

## Mastery gate

Given a reproducible NaN at step 37, identify the first invalid operation, explain whether range, numerical formula, data, or optimization caused it, and make the smallest repair. Produce a table that shows why BF16, FP16/scaled, or FP32 is the justified policy for each workload phase.

## Primary sources, code paths, and OSS bridge

- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/amp.html)
- [Autocast API](https://docs.pytorch.org/docs/stable/amp.html#torch.autocast)
- [Numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [IEEE 754-2019 standard overview](https://standards.ieee.org/ieee/754/6210/)

Read `torch/amp/`, `aten/src/ATen/autocast_mode.*`, and `torch/cuda/amp/` in [PyTorch](https://github.com/pytorch/pytorch). Good first upstream work is a minimal correctness reproducer, a missing autocast test, or documentation that states an operation’s accumulation contract. Do not submit an “optimization” until eager FP32, AMP eager, and compiled paths have a clear correctness matrix.

## Next

Module 23 uses these stability habits while fine-tuning a real language or multimodal model with torchtune.
