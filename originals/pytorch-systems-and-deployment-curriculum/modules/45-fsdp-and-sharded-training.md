# Module 45 — FSDP and Sharded Training

**Phase:** 7 — Distributed systems foundations  
**Time:** 18 focused hours  
**Prerequisites:** Modules 43–44, Modules 16 and 22, and a memory-accounted DDP baseline.

## Existing inference curriculum bridge

This deepens [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md) and connects its memory/communication tradeoffs to [13 — Mixture of Experts](../../inference-engineering-curriculum/modules/13-mixture-of-experts.md), [14 — Long context and memory](../../inference-engineering-curriculum/modules/14-long-context-and-memory.md), and [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md).

## Why this matters and outcomes

FSDP is where “model fits” stops being a static parameter-count calculation. Parameters, gradients, optimizer state, activations, all-gather buffers, allocator fragmentation, and checkpoint format all move through different ownership states. You need that mental model before diagnosing large-model training or evaluating a claim that a sharding recipe is faster, cheaper, or portable to serving.

You will be able to account for sharded state, select a wrapping/mesh strategy from a model structure, distinguish legacy FSDP1 from FSDP2’s `fully_shard` path, checkpoint without rank-zero memory explosions, and design a local validation when one 16 GB GPU cannot honestly exercise a large multi-GPU configuration.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P45 — FSDP and sharded training**.

1. [FSDP API](https://docs.pytorch.org/docs/stable/fsdp.html) — learn the API or system contract.
2. [FSDP advanced tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html) — study the underlying design, paper, or specification.
3. [ZeRO paper](https://arxiv.org/abs/1910.02054) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

For (P) parameter bytes and Adam-like optimizer state, replicated DDP keeps approximately (P + P_{grad} + P_{optim}) per rank before activations. Full sharding distributes persistent parameters, gradients, and optimizer state across (p) ranks, but materializes parameter groups temporarily with all-gathers before compute and commonly uses reduce-scatter after backward. Peak memory therefore depends on wrap granularity, prefetch, activation shapes, and concurrent all-gather lifetimes—not simply (1/p).

PyTorch currently documents both `FullyShardedDataParallel` (often called FSDP1) and FSDP2’s `torch.distributed.fsdp.fully_shard`. FSDP2 is an evolving API built around DTensor/per-parameter sharding; pin a PyTorch version and read the migration guidance before copying a recipe. Treat neither API's defaults as a universal prescription. In FSDP1, optimizer construction follows wrapping because parameter objects change; `use_orig_params=True` is required for `torch.compile` according to current docs. In either version, module boundaries determine communication/computation overlap and checkpoint shape.

Your hardware supports meaningful memory profiling for one-rank reference models, but not a claim about cross-GPU FSDP throughput. Use CPU/Gloo multi-process toy models to validate state/checkpoint semantics, estimate multi-rank bytes analytically, and defer NCCL overlap measurements until unique GPUs are available. Meta-device initialization can help inspect a large architecture without allocating its weights; it is not a substitute for numerical training.

## Implementation ladder in `torch-lab/`

1. **Reference:** make `torch-lab/distributed/state_accounting.py` compute parameters, gradient dtype, optimizer state, activation estimates, and checkpoint bytes for DDP/full-shard assumptions. Validate it on a small Transformer.
2. **Optimized:** implement `torch-lab/distributed/fsdp_toy.py` on two CPU Gloo ranks. Exercise a documented FSDP API version, explicit module wrapping, mixed-precision policy, and state-dict save/load; tag the result as semantic validation.
3. **Integrated:** add `fsdp_plan.md` for a real multi-GPU target: mesh, wrap boundaries, expected all-gathers/reduce-scatters, checkpoint format, recovery budget, and profiling acceptance criteria. Only populate performance fields from actual compatible hardware.

## Three focused sessions

1. **Memory ledger:** derive state bytes for FP32, BF16 parameters with FP32 optimizer state, and several world sizes. Then add an observed `max_memory_allocated` one-GPU reference and explain the difference from the estimate.
2. **Sharding lifecycle:** trace one Transformer block through pre-forward gather, forward reshard, pre-backward gather, gradient reduction, and local optimizer update. Change a wrap boundary and predict the peak-memory/collective change before running.
3. **Checkpoint portability:** save and reload a model/optimizer with a documented state-dict strategy. Test interrupted writes and distinguish a training-sharded checkpoint from an inference-ready full state dict.

## Experiments and deliverables

Commit `state_accounting.py`, `fsdp_toy.py`, `test_fsdp_state_dict.py`, `memory_ledger.csv`, and `fsdp_plan.md`. Record PyTorch/FSDP API version, sharding strategy, module boundaries, parameter/gradient/optimizer dtypes, activation assumptions, rank/device mapping, state-dict type, load destination, peak allocated/reserved memory, communication events, and which entries are measured versus modeled.

## Common failure modes

- Treating sharded persistent state as proof that peak memory is small.
- Creating the optimizer before FSDP wrapping when the selected API changes parameters.
- Saving a full state dictionary on every rank or gathering it accidentally on the host.
- Calling a child module's forward outside the wrapper that manages its shards.
- Conflating FSDP checkpoint layout with a serving checkpoint format.

## Mastery gate

Given a model, dtype policy, world size, and wrap plan, predict its persistent and transient state movement, then verify a toy version’s state-dict round trip. Explain why a proposed configuration fails on 16 GB before running it, and name the evidence required to claim a multi-GPU improvement.

## Primary sources, code paths, and OSS bridge

- [FSDP API and limitations](https://docs.pytorch.org/docs/stable/fsdp.html)
- [FSDP2 `fully_shard` API](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch distributed checkpoint documentation](https://docs.pytorch.org/docs/main/distributed.checkpoint.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)

Read `torch/distributed/fsdp/`, `torch/distributed/_composable/fsdp/`, and TorchTitan's parallelism/checkpoint configuration at the pinned revision you run. Upstream work needs a tightly scoped correctness, state-dict, or memory-regression test; do not submit a broad FSDP rewrite without maintainer direction.

## Next

Module 46 makes meshes and tensor placements explicit instead of burying groups inside wrappers.
