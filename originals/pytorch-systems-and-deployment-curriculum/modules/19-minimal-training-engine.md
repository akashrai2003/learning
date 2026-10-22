# Module 19 — Minimal Training Engine

> **Phase:** II — Reliable PyTorch systems  
> **Time:** 10–12 focused hours across three sessions  
> **Prerequisites:** Modules 10–18 and a tested toy model/dataset

## Existing inference curriculum bridge

Build the training counterpart to [Module 06 — Transformer Forward Pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), [Module 19 — Calibration, Error Metrics, and Evaluation](../../inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md), [Module 25 — FP8 Training and Inference](../../inference-engineering-curriculum/modules/25-fp8-training-and-inference.md), and [Module 44 — Profiling, Benchmarking, and Correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). An inference engineer who can reproduce the path that created weights can diagnose their behavior much better.

## Why this matters and outcomes

Framework recipes can hide the state transition that actually trains a model. A minimal engine is not a toy script; it is the smallest implementation whose data, model, gradients, optimizer, schedule, randomness, checkpoint, metrics, and timing can all be inspected and tested. It becomes the reference implementation before you introduce distributed training, compilers, or finetuning frameworks.

By the end, you can construct a typed training-step state machine; separate pure model/loss from orchestration; support accumulation, validation, metrics, failure handling, and checkpoint boundaries; make a testable resume plan; and measure tokens/examples per second without contaminating the hot path.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P19 — Minimal training engine**.

1. [PyTorch training tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — learn the API or system contract.
2. [Automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html) — study the underlying design, paper, or specification.
3. [PyTorch examples](https://github.com/pytorch/examples) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch examples](https://github.com/pytorch/examples) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch recipes](https://docs.pytorch.org/tutorials/recipes/recipes_index.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and tensor contracts

For a causal language model, a typical batch contract is input_ids: int64[B,T], labels: int64[B,T], attention_mask: bool-or-int[B,T], and logits: float[B,T,V]. Cross entropy reduces valid token losses to scalar L. A train update is not merely model(batch): it is a transition

(theta, optimizer, scheduler, scaler, rng, sampler, counters, batch)
→ (theta_next, optimizer_next, scheduler_next, scaler_next, counters_next, metrics).

Make ownership explicit. The model owns parameters/buffers. The data layer owns CPU samples/collation. The engine owns device movement, module mode, grad mode, loss normalization, accumulation boundary, clipping, update order, logging, and checkpoint invocation. Configuration is immutable resolved input; artifacts are outputs. This separation keeps the model usable for eager evaluation, compiler experiments, export, and serving.

A logical update containing A microbatches should define its normalization. If each microbatch loss is mean-reduced over equal token counts, multiplying or dividing consistently makes the accumulated gradient match the intended large-batch objective. For variable-length token batches, normalize by actual valid-token count rather than blindly by A. Call optimizer.step only after the complete logical update; otherwise momentum/schedulers see a different optimization problem.

Evaluation changes Module mode and uses no_grad or inference_mode according to its downstream contract. It must not update optimizer, scheduler, RNG unintentionally, or training metrics. Metrics should be streamed from sufficient statistics where possible; retaining every logit or output can change memory behavior. CUDA timing must have explicit event/synchronization boundaries, while logging .item or printing tensors must stay outside the measured region.

A small engine should fail early: assert batch device/dtype/shape, finite loss/gradients, parameter coverage, and monotonic counters. On failure, write a diagnostic manifest plus bounded context, not a giant unsafe pickle or hidden retry. Exact resume is defined by Modules 14–16, not assumed.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/training/engine.py with pure model/loss functions and an explicit train_one_update(batch_group, state) interface. Train a deterministic toy classifier on CPU and test one update against a hand/optimizer oracle.
2. Add typed batch/result/state records, finite checks, accurate-token loss normalization, eval mode, and metric accumulators. Test state changes field by field.

**Optimized**

3. Add microbatch accumulation, gradient clipping, optional device transfer policy, and CUDA-event timers without changing the reference objective. Compare accumulated and equivalent concatenated batches.
4. Add a bounded hook interface for logging/checkpoint requests; assert hooks cannot mutate model/optimizer state accidentally. Profile only after the correctness suite passes.

**Integrated**

5. Provide train, evaluate, resume, and benchmark entry points using one resolved configuration/manifest contract. Integrate Module 15 RNG and Module 16 checkpoint inventories.
6. Run a fault-injection suite: NaN loss, wrong token dtype, missing parameter group, data exception, interrupted checkpoint, and resume mismatch. Make each failure actionable.

## Three focused sessions

1. **One correct update (3h):** define contracts, build a CPU oracle, implement model/loss/backward/step, and log a state transition table.
2. **Accumulation and evaluation (3h):** prove variable-length normalization and accumulation parity, then isolate eval from training state.
3. **Operation and failure (3h):** add manifests, checkpoints, timers, and fault tests; produce one clean small-run report.

## Experiments and deliverables

Commit engine.py, state dataclasses, update/evaluation tests, accumulation-parity test, metrics schema, fault matrix, and a training run report. Include model/data revisions, tensor contracts, update count versus microbatch count, tokens/s timing method, peak memory, and exact/approximate resume claim.

## Common failure modes

- Embedding data loading, device movement, loss, and optimizer mutation inside an untestable model forward.
- Dividing loss by microbatch count when valid-token counts differ.
- Stepping optimizer/scheduler on every microbatch unintentionally.
- Letting evaluation mutate train state or save gradients.
- Calling item/synchronize/logging inside the performance interval.
- Retrying NaNs/OOMs without preserving the first failure evidence.

## Mastery gate

From a clean run, demonstrate one update that matches an oracle, accumulated updates that match the stated objective, evaluation with no train-state mutation, and checkpoint/resume behavior with a declared reproducibility scope. Diagnose every fault-injection case from its emitted evidence.

## Primary sources and canonical code paths

- [PyTorch Modules note](https://docs.pytorch.org/docs/stable/notes/modules.html)
- [PyTorch optimizer documentation](https://docs.pytorch.org/docs/stable/optim.html)
- [PyTorch data loading documentation](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler.html)
- Read PyTorch’s module, optimizer, DataLoader, and test harness source trees before borrowing structure.

## OSS bridge and next

A minimal engine gives you a trustworthy oracle for upstream fixes and new kernels. Contributions can start as an isolated correctness test or benchmark reproducer derived from it. Module 20 replaces the toy model with a tested decoder-only Transformer whose tensor and state semantics remain visible.
