# Module 44 — DistributedDataParallel

**Phase:** 7 — Distributed systems foundations  
**Time:** 15 focused hours  
**Prerequisites:** Module 43, Modules 14 and 19, and a tested single-process training loop.

## Existing inference curriculum bridge

Pair this with [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md), [46 — vLLM internals and contributions](../../inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md), and [47 — SGLang internals and contributions](../../inference-engineering-curriculum/modules/47-sglang-internals-and-contributions.md). DDP replicates a training model; serving replicas use a related operational idea but different batching, state, and tail-latency constraints.

## Why this matters and outcomes

DDP is the cleanest way to learn how a framework turns backward gradients into communication without pretending model sharding is free. It establishes the baseline against which FSDP, tensor parallelism, and serving replication should be evaluated. A DDP run that happens to converge is insufficient: you must know whether every sample is seen exactly once, gradients are equivalent to a global-batch reference, and reductions overlap the intended backward computation.

You will be able to build a correct one-process-per-device DDP program, use `DistributedSampler` correctly, reason about bucketed gradient all-reduces, investigate unused parameters, resume with rank-safe checkpoints, and write an honest one-GPU limitation statement.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P44 — DistributedDataParallel**.

1. [DDP API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — learn the API or system contract.
2. [DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html) — study the underlying design, paper, or specification.
3. [PyTorch distributed tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Each rank owns a full model replica and receives a disjoint microbatch. During backward, DDP groups gradients into buckets and all-reduces them; after reduction, every replica applies the same optimizer step. With a per-rank batch (b), world size (p), and gradient accumulation (a), the effective batch is normally (bpa), provided the loss scaling, sampler, and accumulation boundaries agree. Communication volume is approximately the model’s gradient bytes per synchronized step; DDP saves no parameter or optimizer-state memory relative to one replica.

DDP expects one process per GPU for CUDA work. It must be initialized before wrapping, and the optimizer should see the wrapped model’s parameters. `DistributedSampler` needs `set_epoch(epoch)` so ranks reshuffle deterministically but differently each epoch. `no_sync()` is for deliberate gradient accumulation; accidentally placing a normal backward inside it silently changes the global-update contract. `find_unused_parameters=True` is a diagnostic/compatibility tool with cost and semantics, not a default performance flag. `static_graph` is only defensible after proving parameter-use structure is invariant.

On your single RTX 5060 Ti, make the CPU/Gloo two-rank run the correctness oracle and use one-rank CUDA for real local training/profiling. Do not use two ranks sharing one GPU as evidence of DDP speedup; that tests contention and launch behavior, not data-parallel scaling.

## Implementation ladder in `torch-lab/`

1. **Reference:** preserve `torch-lab/train/single_process.py` and add a deterministic synthetic dataset plus a manual global-batch gradient oracle.
2. **Optimized:** create `torch-lab/distributed/ddp_train.py` with Gloo CPU ranks, `DistributedSampler`, rank-zero logging/checkpointing, and a final parameter equality test against the reference for fixed seeds.
3. **Integrated:** add CUDA-aware DDP configuration guarded by `world_size > 1` and available unique GPUs. Capture a PyTorch profiler trace to show backward buckets; when only one GPU exists, mark the scaling section as deferred rather than emulating results.

## Three focused sessions

1. **Equivalence:** train a tiny MLP for one update in single-process and two-rank CPU DDP. Fix the loss normalization and data partitioning until parameters agree within a declared tolerance.
2. **Lifecycle:** add sampler epoch seeding, rank-zero artifacts, all-rank failure propagation, checkpoint resume, and a test that every sample ID appears exactly once per epoch across ranks.
3. **Performance model:** inspect gradient bucket timing, change bucket size only after documenting the baseline, and explain why smaller buckets may improve overlap but add collective overhead.

## Experiments and deliverables

Commit `ddp_train.py`, `test_ddp_equivalence.py`, `test_sampler_coverage.py`, `ddp_config.yaml`, `ddp_trace.json`, and `ddp_report.md`. Include model/optimizer hashes, per-rank seeds and batch sizes, global batch calculation, sampler coverage, loss reduction convention, parameter/optimizer checkpoint ownership, communication backend, and whether an observed number is CPU semantic validation, single-GPU profiling, or genuine multi-GPU data.

## Common failure modes

- Every rank reads the whole shuffled dataset because `DistributedSampler` is missing.
- Forgetting `sampler.set_epoch`, producing repeated shard order.
- Scaling learning rate or loss twice when changing world size.
- Saving concurrent checkpoints from every rank to one path.
- Hiding a dynamically unused parameter with a flag instead of understanding the model branch.

## Mastery gate

Produce a two-rank CPU DDP run whose final model and optimizer state match a specified global-batch reference. Diagnose a deliberately injected sampler, unused-parameter, and checkpoint-race bug from evidence. State exactly what you would need before making a GPU scaling claim.

## Primary sources, code paths, and OSS bridge

- [PyTorch DistributedDataParallel API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed data parallel notes](https://docs.pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch distributed source](https://github.com/pytorch/pytorch/tree/main/torch/nn/parallel)

Read `torch/nn/parallel/distributed.py`, reducer/c10d code reached from it, and `test/distributed/`. Contributions that expose an actual reducer, sampler, or diagnostic regression should include a small reproducer and multi-rank test; avoid changing heuristics from one hardware trace.

## Next

Module 45 trades DDP’s replicated state for sharding and new communication costs.
