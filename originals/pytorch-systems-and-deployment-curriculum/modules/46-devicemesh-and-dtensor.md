# Module 46 — DeviceMesh and DTensor

**Phase:** 7 — Distributed systems foundations  
**Time:** 15 focused hours  
**Prerequisites:** Modules 43 and 45, tensor-stride literacy, and basic SPMD reasoning.

## Existing inference curriculum bridge

Connect this module to [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md), [13 — Mixture of Experts](../../inference-engineering-curriculum/modules/13-mixture-of-experts.md), [37 — GEMM and tensor cores](../../inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md), and [42 — Compiler stack: Dynamo, Inductor, and Triton](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md). The mesh/placement notation makes a parallel GEMM or routed expert exchange inspectable before a runtime hides it.

## Why this matters and outcomes

Manual process-group bookkeeping does not scale to 2-D or 3-D parallelism. `DeviceMesh` describes topology and creates dimension-specific communicators; DTensor attaches a global-tensor intent and placements to local storage. Together they make it possible to reason about “this weight is sharded on the output feature dimension and replicated across data-parallel ranks” instead of “rank 7 did something special.”

You will be able to create a consistent mesh, use `Replicate`, `Shard`, and `Partial` placements, explain redistribution, inspect local versus global tensor metadata, and recognize where DTensor is a programming model rather than a magic performance layer. APIs in this area evolve quickly, so every experiment must pin PyTorch and retain its trace.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P46 — DeviceMesh and DTensor**.

1. [DeviceMesh API](https://docs.pytorch.org/docs/stable/distributed.tensor.html) — learn the API or system contract.
2. [DTensor API](https://docs.pytorch.org/docs/stable/distributed.tensor.html) — study the underlying design, paper, or specification.
3. [Tensor parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

A (d_0 \times d_1) `DeviceMesh` maps global ranks onto named dimensions, such as `("dp", "tp")`. A DTensor has a conceptual global shape plus one placement per mesh dimension. `Replicate()` keeps the complete global value on ranks along that mesh dimension. `Shard(k)` partitions global dimension (k). `Partial()` represents a value awaiting a reduction; it is not interchangeable with a replicated result. Redistribution between placements emits communication—often all-gather, reduce-scatter, or all-reduce—so it is an algorithmic cost, not metadata relabeling.

The essential invariant is SPMD agreement: every rank must create the same mesh layout, names, and operation sequence. A mismatched mesh can hang rather than throw. DTensor's global semantics also do not license arbitrary operations: an operation needs a valid sharding rule, compatible shapes, and appropriate reduction semantics. Inspect the local tensor and placements before trusting an output. Empty shards and uneven dimensions need explicit tests.

On a single GPU, use CPU Gloo ranks for mesh/placement correctness and a world-size-one CUDA run to inspect API behavior. A sequential local simulation helps visualize ownership but cannot prove collective implementation performance. Do not split one 5060 Ti into fake CUDA ranks to benchmark tensor parallelism.

## Implementation ladder in `torch-lab/`

1. **Reference:** build `torch-lab/distributed/placement_simulator.py` that maps a global matrix to local ranges for replicate, row-shard, column-shard, and partial-sum states. Its oracle should work without PyTorch distributed.
2. **Optimized:** implement `dtensor_lab.py` with a CPU `DeviceMesh`, `distribute_tensor`, explicit placements, local-tensor inspection, and redistribution assertions for divisible and non-divisible shapes.
3. **Integrated:** write `linear_placements.md` for a column-parallel and row-parallel linear layer. State which matmul result is partial, where its all-reduce happens, what output layout downstream expects, and how it composes with data-parallel replication.

## Three focused sessions

1. **Mesh construction:** create 1-D and 2-D CPU meshes, print each rank’s coordinate/groups, then deliberately mismatch a mesh under a short timeout to learn the diagnostic signature.
2. **Placement algebra:** hand-calculate local shards for small matrices, construct DTensors, and test round-trip `full_tensor()` only as a correctness oracle—not as a fast path.
3. **Parallel linear design:** derive column and row sharding for (Y=XW\). Mark partial outputs and communication boundaries, then compare the planned tensor layouts to a single-process reference.

## Experiments and deliverables

Commit `placement_simulator.py`, `dtensor_lab.py`, `test_placements.py`, `mesh_manifest.json`, and `linear_placements.md`. Capture global shape, local shapes/offsets, placement tuple, mesh axes/ranks, redistribution operation, expected numeric result, uneven-shard behavior, PyTorch revision, and whether execution used CPU Gloo, one CUDA rank, or real unique GPUs.

## Common failure modes

- Defining a different mesh order or shape on one rank.
- Calling `full_tensor()` in a hot path and mistaking it for a local conversion.
- Treating `Partial` as a valid replicated tensor before reduction.
- Sharding a dimension whose operation has no supported or correct propagation rule.
- Forgetting that a placement change moves bytes and can dominate a small GEMM.

## Mastery gate

Draw and implement a two-dimensional mesh plan for a linear layer with data-parallel replication and tensor-parallel weight shards. Predict each rank’s local shapes and every collective, then pass numeric tests including an uneven global dimension. Explain why this is semantic evidence, not a consumer-GPU scaling benchmark.

## Primary sources, code paths, and OSS bridge

- [PyTorch DeviceMesh and distributed API](https://docs.pytorch.org/docs/stable/distributed.html)
- [DTensor documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [Tensor-parallel APIs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [DeviceMesh recipe](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)

Read `torch/distributed/device_mesh.py`, `torch/distributed/tensor/`, and the matching tests at your pinned revision. Useful contributions are small placement-rule, error-message, or uneven-shard regressions with a minimal mesh test; coordinate semantic changes with maintainers first.

## Next

Module 47 combines mesh ideas into tensor, pipeline, context, and expert parallelism.
