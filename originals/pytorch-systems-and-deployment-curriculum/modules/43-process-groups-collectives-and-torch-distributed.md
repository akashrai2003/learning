# Module 43 — Process Groups, Collectives, and `torch.distributed`

**Phase:** 7 — Distributed systems foundations  
**Time:** 15 focused hours  
**Prerequisites:** Modules 09, 18, and 40; basic networking/process concepts; no second GPU is required.

## Existing inference curriculum bridge

This supplies the framework mechanics behind [36 — CUDA execution, memory, and streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md), and [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md). The inference module explains why a serving engine partitions work; this module makes ranks, communicators, ordering, and collectives concrete.

## Why this matters and outcomes

Distributed PyTorch is not “one Python program using many GPUs.” It is a set of independently scheduled processes that must agree on membership, topology, tensor metadata, collective order, and failure handling. A wrong answer is usually a silent hang, incorrect reduction, or a performance result contaminated by initialization and synchronization. Those are the same failure classes you will see when reading vLLM, SGLang, TorchTitan, or NCCL-heavy native code.

You will be able to launch a reproducible multi-process program, distinguish rank/world size/local rank, choose Gloo versus NCCL for the experiment, build subgroups, reason about all-reduce/all-gather/reduce-scatter/all-to-all, and turn a hang into evidence rather than killing processes at random.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P43 — Process groups, collectives, and torch.distributed**.

1. [torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html) — learn the API or system contract.
2. [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — study the underlying design, paper, or specification.
3. [Distributed communication package source](https://github.com/pytorch/pytorch/tree/main/torch/distributed) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

`torchrun` starts one worker process per requested slot. A rendezvous assigns each worker a global `rank` in `[0, world_size)`. `init_process_group()` creates a default communicator; `new_group()` creates a communicator for a named subset. A `DeviceMesh` later makes these group relationships explicit, but it does not remove the SPMD requirement: every rank must construct compatible groups and issue matching collectives in matching order.

For a tensor of (N) elements over (p) ranks, `all_reduce(sum)` leaves a length-(N) sum on every rank. `all_gather` materializes (pN) logical elements on every rank. `reduce_scatter` reduces then retains roughly (N/p) elements per rank. `all_to_all` changes ownership and is central to MoE routing. The useful question is not only “what output do I get?” but “who holds which bytes before and after, over which links, and where can communication overlap compute?”

Backend selection is a capability and environment contract. Use `gloo` for the first CPU, local-multiprocess labs; use `nccl` only when each rank owns an actual CUDA device and the installed PyTorch/NCCL/driver stack supports it. One RTX 5060 Ti cannot create a meaningful multi-GPU NCCL scaling result. It can still teach process semantics with two CPU ranks and one-rank CUDA correctness. Pin your PyTorch version because distributed APIs and diagnostics evolve.

## Implementation ladder in `torch-lab/`

1. **Reference:** write `torch-lab/distributed/collectives.py`, launched with `torchrun --standalone --nproc-per-node=2`, using Gloo and CPU tensors. Print a structured rank manifest and assert exact results for broadcast, barrier, all-reduce, all-gather, reduce-scatter, and object exchange.
2. **Optimized:** add `torch-lab/distributed/groups.py` with even/odd or 2-D toy subgroups. Measure startup, collective latency, and bytes separately; use `async_op=True` only when the work handle is awaited before consumption.
3. **Integrated:** add a one-rank CUDA variant and a `hang_playbook.md`: ordered rank logs, `TORCH_DISTRIBUTED_DEBUG=DETAIL`, timeouts, stack collection, environment capture, and a clean `destroy_process_group()` path. Do not invent a two-GPU result from one device.

## Three focused sessions

1. **Membership and invariants:** launch two CPU ranks, write out `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, backend, hostname, and tensor metadata. Deliberately make one rank skip a collective, observe the timeout/debug evidence, then repair collective order.
2. **Collective algebra:** predict and test every rank’s output for small integer tensors. Derive the byte movement for each collective and compare it with the API's input/output ownership.
3. **Group design and async discipline:** build two subgroups identically on all ranks, issue an asynchronous operation, demonstrate the incorrect early read, then wait and profile the correct version.

## Experiments and deliverables

Commit `collectives.py`, `groups.py`, `test_collectives.py`, `launch_cpu.sh`, `rank_manifest.jsonl`, and `hang_playbook.md`. The report must include command line, PyTorch/backend versions, rendezvous method, all rank-to-device mappings, timeouts, tensor shapes/dtypes, expected outputs, process exit codes, warm-up policy, and whether timing includes synchronization. Record why a CPU/Gloo trace demonstrates semantics but not GPU communication bandwidth.

## Common failure modes

- Treating `LOCAL_RANK` as a global identity in a multi-node design.
- Calling collectives conditionally on data that differs by rank.
- Using an asynchronous tensor before waiting on its work handle or stream dependency.
- Initializing groups in different order, or destroying only some ranks.
- Reporting a single-GPU process experiment as NCCL scaling.

## Mastery gate

From a failing two-rank trace, identify the first mismatched collective or membership assumption, make the smallest correction, and prove all rank outputs. Draw a byte-ownership diagram for all-reduce, all-gather, reduce-scatter, and all-to-all, then explain which one a later parallelism strategy needs and why.

## Primary sources, code paths, and OSS bridge

- [PyTorch distributed communication package](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch distributed examples](https://github.com/pytorch/examples/tree/main/distributed)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

Read `torch/distributed/`, `torch/csrc/distributed/c10d/`, and tests under `test/distributed/`. A good first contribution is a minimal correctness or diagnostic test with a precise rank transcript—not a new collective abstraction. Verify current maintainership and an actionable issue before investing in a patch.

## Next

Module 44 applies these primitives to replicated training with DistributedDataParallel.
