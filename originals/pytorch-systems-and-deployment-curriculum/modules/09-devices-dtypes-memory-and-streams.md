# Module 09 — Devices, Dtypes, Memory, and Streams

> **Phase:** I — Engineering foundations  
> **Time:** 9–11 focused hours across three sessions  
> **Prerequisites:** Module 08 and basic accelerator terminology

## Existing inference curriculum bridge

Pair this with [Module 04 — Numerical Formats and Error](../../inference-engineering-curriculum/modules/04-numerical-formats-and-error.md), [Module 05 — Performance Modeling and Roofline](../../inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md), and [Module 36 — CUDA Execution, Memory, and Streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md). The prior curriculum explains hardware-level constraints; this module teaches the PyTorch contracts that expose them safely in application code.

## Why this matters and outcomes

A tensor program can be mathematically correct yet slow, nondeterministic, or out of memory because its device, dtype, lifetime, or stream assumptions are wrong. Inference engineering lives at this boundary: model weights, KV-cache pages, staging buffers, CUDA graph pools, and custom kernels rely on the same rules.

By the end, you can specify device/dtype/layout contracts; reason honestly about host-to-device transfers and pinned memory; distinguish live, reserved, and externally visible GPU memory; establish correct stream ordering and allocator lifetimes; and construct a GPU benchmark whose timing means what it claims.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P09 — Devices, dtypes, memory, and streams**.

1. [CUDA semantics in PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html) — learn the API or system contract.
2. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — study the underlying design, paper, or specification.
3. [Understanding CUDA memory usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ATen native source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Every tensor has one device and one dtype. Moving a tensor and casting a tensor are distinct transformations even if one to call requests both. Make the device/dtype policy a boundary concern: parameters, inputs, temporary buffers, scalar constants, index tensors, and output all deserve an explicit contract. An accidental CPU tensor or a silent promotion can cause a copy, synchronization, or different kernel. Float32, float16, and bfloat16 have different range/precision tradeoffs; an input dtype alone does not define the operator’s accumulation dtype.

CUDA work is queued asynchronously on a stream. Python can return after enqueueing a kernel but before the GPU completes it. Operations in one stream retain FIFO order. Operations across streams need an explicit dependency. For a producer on stream A and consumer on stream B, record an event after producing, make B wait for it, then launch consumption on B. If a tensor’s storage is used on a side stream, call tensor.record_stream(stream_b) so PyTorch’s caching allocator will not reuse that storage early. The event establishes execution order; record_stream establishes storage lifetime. They are different guarantees.

Synchronize only at a host-observation or benchmark boundary. CUDA events can measure device elapsed time without timing Python dispatch; a warm benchmark must exclude initialization, compilation, and autotuning according to its written policy. A default stream is not a global proof that every extension or library shares your intended dependency graph.

The caching allocator keeps freed blocks for fast future allocation. memory_allocated refers to live payload managed by the allocator; memory_reserved includes cached blocks held by it; nvidia-smi can additionally show contexts, libraries, and allocations outside that view. empty_cache releases unoccupied cached blocks to the driver. It cannot free live tensors, fix a leak, or make a genuinely too-large model fit. Diagnose OOM by tracing object lifetime, shape/batch growth, peak allocation, and allocator statistics.

Pinned CPU memory is page-locked host memory suitable for efficient DMA. A DataLoader with pin_memory=True can prepare batches, while to(device, non_blocking=True) may let the host continue when source, stream, and lifetime conditions permit. This is not a universal speed flag: pinned memory is limited host resource, and later use can introduce a necessary wait.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/systems/tensor_inventory.py. For every boundary tensor, report shape, dtype, device, strides, element bytes, payload bytes, and current/default stream.
2. Build a CPU-to-GPU transfer benchmark that compares pageable and pinned batches. Record host wall-clock and CUDA-event time separately, synchronizing only after a trial.

**Optimized**

3. Write a two-stream producer/consumer experiment. First make the missing event race observable, then repair ordering and storage lifetime with an event and record_stream.
4. Write an allocation report around toy decoder prefill and decode: allocated, reserved, peak statistics, and, where available, a memory snapshot. Attribute large allocations to weights, activations, workspace, or KV cache.

**Integrated**

5. Repeat selected operators in FP32, BF16, and FP16 only when supported by the target GPU. Compare both numerical error against FP32 and selected-kernel/profiler evidence.

Keep CPU execution as a correctness baseline; design device policy so CUDA, other accelerator backends, and CPU tests fail clearly rather than silently choosing defaults.

## Three focused sessions

1. **Contracts and transfers (3h):** introduce device/dtype assertions at model boundaries. Benchmark pageable and pinned transfers, then document exactly which point makes each measurement complete.
2. **Allocator forensics (3h):** make a controlled allocation/lifetime workload, reset peak stats, compare allocated/reserved/nvidia-smi, and show why empty_cache changes only part of the picture.
3. **Streams and timing (3h):** construct explicit producer/consumer streams and CUDA events. Demonstrate broken order, repair it, then measure a representative GEMM with a defensible warm-up protocol.

## Experiments and deliverables

Commit a transfer report, stream-timeline diagram, allocation report, and benchmark JSON with GPU, driver, PyTorch build, dtype, shape, warm-up policy, raw samples, median, and dispersion. Include one intentionally broken synchronization experiment and the smallest repair.

## Common failure modes

- Synchronizing every iteration and calling the result pipeline throughput.
- Assuming non_blocking always produces asynchronous end-to-end transfer.
- Treating empty_cache as a memory optimization rather than an allocator observation.
- Using side streams without both ordering and lifetime management.
- Comparing low-precision modes without testing numerical deviation and actual selected kernels.
- Creating tensors on the unintended default GPU or mixing device guards.

## Mastery gate

Explain a run with high reserved memory but lower allocated memory. Build a two-stream transfer/compute pipeline that passes a race-sensitive test, then demonstrate whether it actually overlaps on your GPU. Defend every timing boundary in a benchmark review.

## Primary sources and canonical code paths

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [PyTorch torch.cuda API](https://docs.pytorch.org/docs/stable/cuda.html)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- Read [CUDACachingAllocator.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp), [CUDAStream.cpp](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAStream.cpp), and torch/csrc/cuda.

## OSS bridge and next

A credible PyTorch issue or contribution starts with a small stream/lifetime reproducer, exact versions, and profiler or allocator evidence. Do not propose a synchronization workaround until you can state what dependency it repairs. Module 10 adds autograd graph and gradient lifetimes to the same systems picture.
