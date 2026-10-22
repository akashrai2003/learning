# Module 18 — PyTorch Profiler, Memory, and Diagnostics

> **Phase:** II — Reliable PyTorch systems  
> **Time:** 10–12 focused hours across three sessions  
> **Prerequisites:** Modules 05, 09, 17 and a repeatable CPU/GPU workload

## Existing inference curriculum bridge

This operationalizes [Module 05 — Performance Modeling and Roofline](../../inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md), [Module 36 — CUDA Execution, Memory, and Streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), [Module 38 — Tiling, Pipelining, and Fusion](../../inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md), and [Module 44 — Profiling, Benchmarking, and Correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). You will learn which instrument can answer which systems question.

## Why this matters and outcomes

A profiler trace is evidence, not an explanation by itself. PyTorch operator time, CUDA kernel time, allocator history, driver-visible memory, CPU gaps, and per-kernel hardware counters are different measurements with different blind spots. Choosing the wrong one wastes days and can change the workload enough to hide the issue.

By the end, you can formulate a falsifiable performance/memory question; capture a bounded PyTorch profiler trace; read allocator snapshots without mistaking them for all VRAM; escalate to Nsight Systems or Nsight Compute appropriately; and produce a diagnosis with a baseline, hypothesis, validation, and correctness guard.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P18 — PyTorch profiler, memory, and diagnostics**.

1. [Profiler API](https://docs.pytorch.org/docs/stable/profiler.html) — learn the API or system contract.
2. [CUDA memory snapshot](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — study the underlying design, paper, or specification.
3. [Nsight Systems](https://docs.nvidia.com/nsight-systems/) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch examples](https://github.com/pytorch/examples) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch recipes](https://docs.pytorch.org/tutorials/recipes/recipes_index.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and diagnostic map

Start with a symptom and choose the narrowest instrument:

| Question | Primary instrument | What it cannot establish alone |
|---|---|---|
| Which PyTorch operators/call sites dominate? | torch.profiler | Exact SM utilization or all non-PyTorch GPU allocations |
| Why is PyTorch allocator reserved memory fragmented or OOM? | CUDA memory snapshot/history | NCCL/direct-CUDA/external allocations; full execution timing |
| Are CPU, CUDA runtime, copies, streams, and kernels leaving gaps? | Nsight Systems | Detailed instruction-level kernel bottleneck |
| Is one kernel compute, memory, occupancy, or instruction limited? | Nsight Compute | End-to-end multi-process scheduling and application semantics |
| What total memory/process utilization is visible to driver? | nvidia-smi/NVML | Tensor/operator ownership and allocator stack traces |

torch.profiler records selected CPU/CUDA activities and can group operator events by name, shapes, stack, or module context. Use a schedule with wait, warmup, active, and profiler.step rather than tracing an entire training run. record_shapes, with_stack, profile_memory, FLOP estimates, and trace exports add overhead; shape recording can hold tensor references and change memory behavior. Therefore profile a representative warmed window, compare it to an unprofiled baseline, and avoid using profiled throughput as the final performance number.

Allocator statistics report live allocated and cached reserved memory. A CUDA memory snapshot records allocator segments, blocks, and optionally event history/stack traces. Enable torch.cuda.memory._record_memory_history only around the issue, bound max_entries, and dump the snapshot for the memory visualizer. It sees memory managed through PyTorch’s allocator. Direct CUDA allocations, communication libraries such as NCCL, some external runtimes, and driver/context memory can be invisible; reconcile unexplained memory with device-level counters rather than declaring a leak from a partial view.

Nsight Systems provides a timeline across CPU threads, CUDA APIs, streams, kernels, copies, and synchronization. It answers whether the GPU is idle due to input, launch, dependency, or transfer gaps. Nsight Compute collects detailed metrics for selected kernels and has higher collection cost; use it after a Systems/profiler trace isolates one kernel. Both perturb execution and need a small deterministic command. GPU timing still needs correct CUDA-event/synchronization methodology from Module 09.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/diagnostics/profile_toy_step.py. Capture a CPU-only and CUDA-optional PyTorch profiler trace around one warmed training/inference step with a declared schedule.
2. Export a trace and operator table; annotate self versus total time, input shapes, call stacks, and the difference from unprofiled wall/CUDA-event measurements.

**Optimized**

3. Add scoped record_function ranges around data, forward, loss, backward, optimizer, prefill, and decode. Use the trace to test one hypothesis such as avoidable layout copy or CPU-side batching gap.
4. Enable allocator memory history only around a controlled OOM/peak case, dump a snapshot, and relate active blocks to tensors/lifetimes. Bound snapshot history and artifact size.

**Integrated**

5. Capture an Nsight Systems timeline for the same minimized command and reconcile CPU gaps, launches, streams, copies, and kernels with the PyTorch trace.
6. Select one verified hot kernel for Nsight Compute. Compare achieved behavior with roofline expectations, then make one isolated change with parity and benchmark tests.

## Three focused sessions

1. **Profiler discipline (3h):** define a question, establish an unprofiled baseline, use schedules/ranges, and identify a trace artifact caused by instrumentation.
2. **Memory forensics (3h):** contrast allocated/reserved/driver memory, capture an allocator snapshot around an intentional peak or OOM, and list blind spots.
3. **Native escalation (3h):** use Systems to explain a gap, then Compute for one kernel only. Produce a before/after evidence chain, not a screenshot collection.

## Experiments and deliverables

Commit a diagnostic decision tree, trace metadata, operator table, memory report, one snapshot interpretation, and a concise Nsight investigation. Each report must contain command, environment, inputs, warm-up, profiler settings, overhead comparison, hypothesis, conclusion, and a correctness test.

## Common failure modes

- Profiling a cold compile/JIT path and calling it steady-state inference.
- Reading self CPU time as GPU kernel time.
- Enabling shapes/stacks/memory history forever and changing the workload under study.
- Treating an allocator snapshot as total device memory.
- Using Nsight Compute before isolating a kernel or profiling an entire server.
- Declaring a speedup without parity, variance, or synchronized measurement.

## Mastery gate

Given one slow step and one OOM, choose instruments in order, state each tool’s blind spots, and produce an evidence chain that isolates a root cause. Demonstrate a trace-guided improvement that preserves numerical results and survives an unprofiled benchmark.

## Primary sources and canonical code paths

- [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch profiler tutorial](https://docs.pytorch.org/tutorials/beginner/profiler.html)
- [PyTorch CUDA memory usage and snapshots](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [PyTorch CUDA memory API](https://docs.pytorch.org/docs/stable/cuda.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- Inspect [torch/profiler](https://github.com/pytorch/pytorch/tree/main/torch/profiler), [CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp), and Kineto integration code.

## OSS bridge and next

Good contributions include a minimized profiler regression, missing profiling range/test, allocator diagnostic improvement, or an evidence-backed performance bug. Never attach customer data or unrestricted traces. Module 19 uses this observability to build a small training engine whose state, timing, and failure boundaries are explicit.
