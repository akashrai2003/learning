# Module 36 — CUDA Execution, Memory, and Streams

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 18, paired with Module 35; three 3-hour sessions plus a CUDA correctness audit  
**Prerequisites:** Module 35 and working C++/CUDA compilation; thread hierarchy, SM resources, PTX/SASS targets, and disciplined timing

## Why this matters

CUDA launches are asynchronous. A program can print a plausible time before its kernel finishes, read data before a producer completes, or accidentally serialize all streams. Memory errors may surface at a later API call, making the innocent call look guilty. In an inference engine, streams, events, allocators, graph capture, and framework ownership form one dependency graph; correctness and performance require making that graph explicit.

## Outcomes

You can:

- index and bound a grid-stride CUDA kernel;
- reason about coalescing, alignment, shared-memory reuse, bank conflicts, and spills;
- express same-stream order and cross-stream dependencies with events;
- time asynchronous kernels correctly and detect implicit synchronization;
- connect framework streams and memory pools to custom/JIT operators.

## Mental model: queues plus happens-before edges

A stream is an ordered device-work queue. Operations in one stream execute in issue order. Operations in different streams may overlap when dependencies and hardware resources allow; different streams do not promise overlap. A recorded event creates a completion marker, and `cudaStreamWaitEvent` adds a device-side dependency without blocking the host. `cudaDeviceSynchronize` waits for all prior device work and is usually too broad for a critical path.

For a one-dimensional grid-stride loop,

\[
i=\text{blockIdx.x}\cdot\text{blockDim.x}+\text{threadIdx.x},
\qquad
i\mathrel{+}=\text{blockDim.x}\cdot\text{gridDim.x}.
\]

Every access still needs a bound. Adjacent lanes should access adjacent suitably aligned elements so transactions coalesce. Reuse may justify loading a tile from global memory into shared memory, synchronizing the block, and computing many times. Padding can remove shared-memory bank conflicts. Too many live values spill registers into local memory, which is physically backed by device memory and can turn a compute kernel into a memory-bound one.

Transfers involving pageable host memory can stage and serialize. Pinned host buffers enable genuine asynchronous host-device copies, but excessive pinning harms the system. Unified memory offers convenient addressing, not free locality; migration and page faults must be profiled. Device allocations are also synchronization-sensitive: `cudaMallocAsync`/`cudaFreeAsync` and memory pools make lifetime stream-ordered, provided cross-stream uses are protected.

Timing follows the dependency graph. CPU wall time around a launch measures enqueue overhead unless synchronized. CUDA events recorded in the measured stream report device elapsed time; warm-up and independent repetitions remain necessary. Debugging may temporarily synchronize after each launch, but that changes execution and should not remain in performance results.

## From the SM120 build to execution

The successful RTX 5060 Ti FlashInfer build established compilation for CC12.0, not correct scheduling. A JIT-created `.so` can still launch on PyTorch's wrong current stream, use a temporary after its allocator lifetime, or trigger first-use compilation inside a timed request. Check that the custom operator obtains the current CUDA stream rather than assuming stream zero, records external-stream uses when required by the framework allocator, and separates cold JIT/cache population from steady state.

CUDA graphs capture a mostly static sequence of operations and replay it with lower host-launch overhead. Capture imposes stable addresses and capture-safe allocation/control flow. Graph pools and multiple batch-shape captures consume memory. On the 16 GB 5060 Ti, that memory cost can be the difference between a successful compile and an OOM during warm-up; graph count is a capacity decision, not merely a speed switch.

## Implementation ladder

### 1. Reference

Create `inference-lab/kernels/cuda/execution_basics.cu` with bounded vector add, grid-stride reduction, and explicit error checks. Add CPU or PyTorch oracles, zero-length and tail sizes, and compute-sanitizer runs.

### 2. Optimized

Build coalesced versus strided copies, shared-memory tiled reuse, padded/unpadded bank-conflict cases, and pinned asynchronous transfers. Sweep blocks and streams. Use events for device timing and Nsight Systems to test whether overlap actually occurs.

### 3. Integrated

Wrap a kernel for PyTorch, launch on its current stream, allocate through compatible mechanisms, and exercise it between producer and consumer work on two streams. Then capture a stable shape in a CUDA graph and compare cold launch, warm eager, and graph replay without changing numerics.

## Three 3-hour sessions

1. **Execution correctness:** implement indexing and tails; observe asynchronous error reporting, then use launch checks and Compute Sanitizer.
2. **Memory movement:** sweep contiguous/strided access, alignment, shared tiling, bank padding, pinned copies, and unified-memory residency.
3. **Streams and graphs:** build an event dependency DAG, inspect overlap in a timeline, integrate the current framework stream, and capture/replay one graph.

## Experiments and deliverables

Commit:

- `kernels/cuda/execution_basics.cu`;
- `kernels/cuda/stream_pipeline.cu`;
- `tests/test_cuda_stream_correctness.py`;
- `benchmarks/results/cuda-memory-streams.json`;
- `notes/profiles/cuda-overlap.md`;
- `notes/failure_postmortems/async-error-attribution.md`.

Record allocation kind, transfer direction, pinned/pageable status, bytes, strides, block/grid, streams, event placement, synchronization, cache state, cold/warm classification, effective bandwidth, profiler timeline, and correctness hash.

## Common failure modes

- Timing only the asynchronous launch.
- Calling device-wide synchronization inside the measured loop.
- Assuming two streams necessarily overlap.
- Using pageable host memory and expecting asynchronous transfer.
- Omitting an event between cross-stream producer and consumer.
- Launching a custom op on the legacy default stream instead of the framework's current stream.
- Freeing or reusing storage before all streams finish with it.
- Diagnosing the API call that reports a prior asynchronous error.
- Ignoring tails, alignment, shared-memory bank conflicts, or register spills.
- Mixing first JIT/graph capture with steady-state throughput.

## Mastery gate

Draw a happens-before graph for host-to-device copy, two kernels on different streams, and device-to-host copy; implement it without a device-wide barrier and prove correct output over repeated runs. Produce an Nsight Systems trace showing overlap or explain from resource/dependency evidence why none occurs. Demonstrate one misleading unsynchronized timing and correct it with events.

## Canonical code areas and OSS bridge

Use the [CUDA samples](https://github.com/NVIDIA/cuda-samples) for idiomatic runtime checks. In PyTorch, inspect `c10/cuda/` stream and allocator code plus C++ extension examples. In vLLM, trace worker/model-runner stream use, CUDA graph capture, and custom-op launch boundaries. In FlashInfer, inspect JIT bindings and launch code around the SM120 operator. Good OSS work includes missing stream tests, capture-safety fixes, lifetime diagnostics, and small reproducible race cases.

## Primary and authoritative sources

- [CUDA Programming Guide: asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA Programming Guide: stream-ordered allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/)
- [Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/)

## Next

Module 37 uses these execution and memory primitives to construct GEMM tiles and reach Tensor Cores with evidence from shapes, instructions, and rooflines.
