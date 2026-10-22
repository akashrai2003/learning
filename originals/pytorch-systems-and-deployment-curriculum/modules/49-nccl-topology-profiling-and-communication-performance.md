# Module 49 — NCCL, Topology, Profiling, and Communication Performance

**Phase:** 7 — Distributed systems foundations  
**Time:** 15 focused hours  
**Prerequisites:** Modules 18 and 43–48, GPU profiling basics, and a written performance hypothesis.

## Existing inference curriculum bridge

This extends [35 — GPU architecture and Blackwell](../../inference-engineering-curriculum/modules/35-gpu-architecture-and-blackwell.md), [36 — CUDA execution, memory, and streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md), and [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md). It is the bridge from a collective's abstract semantics to a topology-bound latency/bandwidth measurement.

## Why this matters and outcomes

Distributed performance fails when engineers use a peak interconnect number as if every collective, shape, process placement, and link gets it. NCCL selects algorithms/protocols against topology and message size; application code controls bucketization, stream dependencies, and overlap. You need a trace that separates compute, communication, host launch gaps, and synchronization before tuning any knob.

You will be able to inventory a topology, build a collective microbenchmark with correct synchronization, capture PyTorch/Nsight evidence, calculate effective algorithmic bandwidth, diagnose common hangs, and say what cannot be measured on one GPU.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P49 — NCCL topology, profiling, and communication performance**.

1. [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — learn the API or system contract.
2. [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) — study the underlying design, paper, or specification.
3. [Nsight Systems](https://docs.nvidia.com/nsight-systems/) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

The physical path can involve GPU PCIe roots, NVLink/NVSwitch, NICs, CPU sockets, NUMA memory, and a process's assigned device. `nvidia-smi topo -m` is an inventory, not a performance result. A ring all-reduce moves each rank's chunk through peers; a tree changes latency/bandwidth tradeoffs. For (N) input bytes/rank and (p) ranks, an all-reduce's algorithmic data movement is often summarized near (2(p-1)N/p), but effective bandwidth must declare the convention, message size, timing scope, and collective implementation.

NCCL calls enqueue GPU work; correct timing brackets work with CUDA events or an intentional synchronization after warm-up. A profiler trace should identify the host-side launch, CUDA streams, NCCL kernels, competing kernels, and waits. Overlap is real only if communication's critical-path time is hidden by useful compute; seeing concurrent colored bars is not enough. `NCCL_DEBUG=INFO` or subsystem logs and `TORCH_DISTRIBUTED_DEBUG=DETAIL` are diagnostics to enable in a bounded reproduction, not default benchmark flags.

Your RTX 5060 Ti exposes one local device. Capture its topology and one-rank CUDA profiler baseline, use CPU/Gloo for semantic collective code, and create a deferred multi-GPU benchmark plan. Do not report NCCL bandwidth, P2P behavior, or overlap from two processes fighting over the same GPU.

## Implementation ladder in `torch-lab/`

1. **Reference:** write `torch-lab/communication/topology_inventory.sh` and `collective_model.py`. Save driver/PyTorch/NCCL versions, topology output, CPU/NUMA data, and predicted bytes for DDP/FSDP/TP/MoE cases.
2. **Optimized:** add `collective_bench.py` that takes backend, ranks, operation, dtype, size sweep, warm-ups, iterations, and timer mode. It must verify results and emit raw samples rather than only a best number.
3. **Integrated:** create `communication_trace.md` containing a PyTorch profiler/Nsight Systems capture, a critical-path annotation, and an experiment plan for a real multi-GPU host. Keep the single-GPU section explicitly labeled baseline/deferred.

## Three focused sessions

1. **Topology and model:** inventory the local host, estimate payload/counts from Modules 44–47, and identify the unanswered topology questions a future multi-GPU host must answer.
2. **Benchmark correctness:** validate warm-up, device synchronization, result equality, and raw percentiles on CPU/Gloo; make a one-rank CUDA timing harness solely to practice event timing.
3. **Trace interpretation:** annotate host gaps, compute kernels, collectives, and blocking waits. Propose one change only if the trace supports its causal mechanism, then rerun the same workload distribution.

## Experiments and deliverables

Commit `topology_inventory.sh`, `collective_model.py`, `collective_bench.py`, `raw_samples.csv`, and `communication_trace.md`. Include hardware/link inventory, GPU mapping, kernel/driver/PyTorch/NCCL versions, message shapes/bytes, ranks, warm-up/iteration count, timer boundaries, median/p95, errors, process affinity, trace configuration, and measured-versus-modeled/deferred labels.

## Common failure modes

- Timing asynchronous launches without a device event or synchronization boundary.
- Comparing different payloads or all-reduce conventions under one “GB/s” label.
- Turning on verbose NCCL logging and calling the perturbed run a benchmark.
- Tuning environment variables before identifying the bottleneck in a trace.
- Confusing a PCIe route shown in topology with actual collective throughput.

## Mastery gate

Given a trace and a collective-size sweep, identify whether the limiting evidence is host launch, compute, communication latency, bandwidth, or synchronization. Produce a deferred multi-GPU plan that names the needed topology, payload, trace, and acceptance threshold instead of inventing a local scaling result.

## Primary sources, code paths, and OSS bridge

- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [PyTorch distributed communication package](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html)
- [Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/)

Read `torch/csrc/distributed/c10d/`, `torch/distributed/`, and the NCCL test/benchmark documentation at the exact release you use. Useful OSS work is a reproducible trace, an incorrect metric/unit fix, or a regression test—never a topology claim based on one consumer GPU.

## Next

Module 50 maps these training parallelisms to serving systems without falsely equating their workloads.
