# Module 47 — Tensor, Pipeline, Context, and Expert Parallelism

**Phase:** 7 — Distributed systems foundations  
**Time:** 18 focused hours  
**Prerequisites:** Modules 43–46, Transformer forward/pass and MoE concepts, and the ability to read a collective trace.

## Existing inference curriculum bridge

Study this alongside [13 — Mixture of Experts](../../inference-engineering-curriculum/modules/13-mixture-of-experts.md), [14 — Long context and memory](../../inference-engineering-curriculum/modules/14-long-context-and-memory.md), [29 — Attention I/O complexity and online softmax](../../inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md), and [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md). It turns the architecture-level terms TP, PP, CP, and EP into ownership and communication diagrams.

## Why this matters and outcomes

“Use 3-D parallelism” is not a configuration. It is a claim about which tensors live on which ranks, which calculation is duplicated, which collective is ordered where, and whether compute can hide the transfer. This module gives you the vocabulary and first implementations to inspect projects such as TorchTitan rather than treating their YAML as a spell.

You will be able to derive a tensor-parallel linear layer, quantify pipeline bubbles, explain context-parallel attention data movement, trace all-to-all expert exchange, choose a feasible composition for a target topology, and state why an implementation is alpha/experimental when the framework says so.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P47 — Tensor, pipeline, context, and expert parallelism**.

1. [PyTorch tensor parallel API](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) — learn the API or system contract.
2. [Pipeline parallel API](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) — study the underlying design, paper, or specification.
3. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Tensor parallelism splits an individual layer. For a column-parallel (Y=XW\), shard (W) by output features; each rank produces a distinct output slice, commonly consumed without immediate communication. A row-parallel next layer shards the input features; its partial output requires an all-reduce. The cost/benefit depends on GEMM dimensions, collective bandwidth/latency, and the activation layout—not a label called TP.

Pipeline parallelism assigns contiguous layers to stages and sends activations/gradients across boundaries. With (m) microbatches and (s) equally timed stages, a simple schedule has bubble exposure roughly proportional to ((s-1)/(m+s-1)\); real imbalance, communication, and memory change it. Current `torch.distributed.pipelining` documentation labels the package alpha, so pin its version and keep a schedule diagram with every experiment.

Context/sequence parallelism partitions token-related work to reduce activation/attention pressure, but attention needs keys/values or partial statistics from other partitions. It is not synonymous with “long context works.” Expert parallelism shards experts; router assignments commonly require all-to-all of token activations, then a reverse exchange. Capacity, padding/drop policy, routing skew, and collective ordering are algorithmic correctness concerns, not just throughput tuning.

Data parallelism can compose with TP/PP/CP/EP on named mesh axes, but each axis must match physical topology. Keep high-volume, frequent collectives inside the fastest local fabric where possible; use pipeline boundaries or replication more carefully over slower links. On one RTX 5060 Ti, derive and simulate plans, run CPU multi-process semantic labs, and use sequential stage calls for functional validation. Defer topology and overlap conclusions until unique devices are available.

## Implementation ladder in `torch-lab/`

1. **Reference:** make `torch-lab/parallelism/ownership_simulator.py`. Given model dimensions and a mesh, emit per-rank parameter/activation ownership, collective names, byte estimates, and a pipeline schedule timeline.
2. **Optimized:** implement CPU/Gloo toy column/row parallel linear layers and a two-expert routed MLP. Compare to a dense single-process oracle; log the explicit all-reduce/all-to-all boundaries.
3. **Integrated:** create `parallelism_design.md` for one compact decoder and a hypothetical multi-GPU cluster. Include mesh axes, layer placement, KV/cache policy, global batch/microbatch count, communication paths, memory ledger, fault domain, and a TorchTitan configuration/source-walk at a pinned commit.

## Three focused sessions

1. **Tensor parallel algebra:** hand-derive column/row sharding of two consecutive linears, then implement the CPU toy and verify every rank's local result and final output.
2. **Pipeline schedule:** draw GPipe and 1F1B timelines for several stage/microbatch counts. Simulate uneven stage times and identify whether the bottleneck is a bubble, a slow layer, or a link.
3. **MoE/context systems:** route tokens through two CPU experts with explicit exchange; build a second worksheet for a context-parallel attention layer that marks what statistics/blocks must cross ranks.

## Experiments and deliverables

Commit `ownership_simulator.py`, `tp_linear.py`, `expert_exchange.py`, `test_parallel_oracles.py`, `pipeline_timeline.svg` or Markdown table, and `parallelism_design.md`. For each result record mesh order, ranks, matrix/token dimensions, local shapes, collective payload/count, microbatch schedule, router distribution/capacity, model/PyTorch revision, and whether it is a simulation, CPU semantic test, one-GPU functional test, or real multi-GPU measurement.

## Common failure modes

- Adding an all-reduce after a column-parallel layer that intentionally keeps output shards.
- Calling a pipeline plan balanced because layer count is equal while measured stage times differ.
- Treating all-to-all as an invisible MoE implementation detail.
- Combining mesh axes whose process groups overlap inconsistently.
- Equating context parallelism with a universal reduction in attention FLOPs or latency.

## Mastery gate

Explain every parameter, activation, and collective for a two-layer TP MLP and a two-stage pipeline. Produce an MoE routing test that agrees with a dense reference. Given a topology and workload, reject one infeasible 3-D plan with a memory or communication argument before benchmarking it.

## Primary sources, code paths, and OSS bridge

- [PyTorch tensor-parallel APIs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [PyTorch pipeline parallelism](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [PyTorch distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

Read `torch/distributed/tensor/parallel/`, `torch/distributed/pipelining/`, TorchTitan's `torchtitan/` parallelism code, and the exact configuration used by a benchmark. Good contributions usually begin with an invariant, an uneven-shape, or a scheduler regression rather than a new megascale feature.

## Next

Module 48 makes state and recovery part of the parallelism design rather than an afterthought.
