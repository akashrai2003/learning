# Module 41 — CUDA Graphs, Memory, and Framework Integration

**Phase:** 6 — Native PyTorch and GPU integration  
**Time:** 12 focused hours  
**Prerequisites:** CUDA streams/allocator, Modules 30–31 and 36–40, and a stable-shape workload.

## Existing inference curriculum bridge

Pair with [12 — KV cache](../../inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md), [14 — Long context/memory](../../inference-engineering-curriculum/modules/14-long-context-and-memory.md), [31 — PagedAttention/memory management](../../inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md), [34 — Continuous batching](../../inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md), [36 — CUDA execution/memory/streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), and [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md).

## Why this matters and outcomes

CUDA Graphs trade dynamism for lower CPU launch overhead: capture a stable sequence of GPU work once, then replay it. This can transform small-batch decode latency, but fixed addresses/shapes/control flow, graph memory pools, and framework allocation rules make it easy to create memory regressions or stale-output bugs. The right question is not “can I enable graphs?” but “which traffic class has a stable replay contract and enough launch overhead to justify the memory/complexity cost?”

You will capture/replay a small PyTorch callable safely, distinguish allocated from reserved memory, reason about graph pools/static buffers, test dynamic-shape limits, and integrate graph policy with compiler and serving shape buckets.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P41 — CUDA graphs, memory, and framework integration**.

1. [PyTorch CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#constraints) — learn the API or system contract.
2. [CUDA Graphs guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) — study the underlying design, paper, or specification.
3. [CUDA memory snapshot](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch custom-ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch extension-cpp examples](https://github.com/pytorch/extension-cpp) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

A CUDA Graph records launches and dependencies as a graph executable. Replay avoids issuing each launch from the CPU, but captured kernel arguments/dependencies and CUDA memory addresses are fixed; values stored at those addresses may change. The common pattern allocates static input/output buffers, copies each new request into static inputs, calls `replay()`, and consumes/clones outputs before the next replay overwrites them. Shapes, strides/layouts, device, and control path must match the captured contract.

Capture has allocation and stream restrictions. PyTorch’s `torch.cuda.graph`/`CUDAGraph` helpers coordinate with the caching allocator, typically using dedicated graph memory pools. These pools can increase total reserved memory because eager and graph allocations may not share freely; newer graph-tree mechanisms can reduce duplication under compatible invocation order. Custom CUDA code launched outside PyTorch’s tracked mechanisms needs especially careful stream/allocation/liveness testing. Do not change capture error mode casually to suppress an unsafe allocation/synchronization error.

Memory accounting needs several views. `memory_allocated` reflects active tensor memory tracked by PyTorch’s allocator; `memory_reserved` is memory held by its caching allocator; driver-visible consumption can include CUDA context, libraries, graph pools, non-PyTorch allocations, and other processes. A fragmentation symptom is not proof that `empty_cache()` is a solution. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can be a useful version-specific experiment for variable allocation patterns, but it neither increases VRAM capacity nor repairs a graph contract/lifetime bug. Record allocator backend/config before interpreting a result.

`torch.compile`/Inductor may use CUDA graph integration for suitable stable CUDA regions in certain modes/configurations. Let framework diagnostics show eligibility and capture sizes. In a server, capture a small set of static decode/prefill/batch shapes, use padding/masks/buckets from Module 31, and preserve an eager/general path for rare or variable requests. Capturing every possible shape can consume more memory than it saves in launch overhead.

## Implementation ladder in `torch-lab/`

1. **Reference:** implement `torch-lab/cudagraphs/eager_baseline.py` with a stable small decoder/MLP call, explicit stream synchronization only for timing, and per-stage allocator snapshots.
2. **Optimized:** add `capture_replay.py` using static inputs/outputs and a `torch.cuda.CUDAGraph` or high-level helper. Verify each replay sees new input values and that previous outputs are preserved when required. Measure capture cost, warm replay latency, allocator reserved/allocated bytes, and graph-pool behavior.
3. **Integrated:** combine compile and graph modes for a bounded set of serving shapes. Implement `graph_cache.py` keyed by true capture contract (shape, dtype, layout, device, relevant mode), a memory budget, eviction policy, and eager fallback. Compare p50/p99 at batch 1 and a realistic batch distribution.

## Three focused sessions

1. **Replay semantics:** capture one pure block, vary values without changing addresses, then intentionally violate shape/address/output-liveness assumptions and explain outcomes.
2. **Memory forensics:** compare eager, one graph, and multiple graph pools using allocator stats/snapshots and driver observations. Test one allocator configuration only after a baseline.
3. **Serving policy:** use a shape histogram to choose capture buckets, account for padding/KV/cache memory, and measure cold capture plus warm p99 versus an eager path.

## Experiments and deliverables

Commit `graph_contract.md`, `capture_replay.py`, `test_replay_values.py`, `test_graph_boundaries.py`, `memory_snapshots/`, `graph_cache_policy.yaml`, and `latency_memory_matrix.csv`. Include graph/compile versions/options, capture stream/pool, static buffer specs, supported shapes/strides/dtypes, capture time, replay time, allocated/reserved/driver memory, p50/p99, output parity, and fallback events.

## Common failure modes

- Replaying without updating static inputs or retaining outputs that replay overwrites.
- Treating a graph capture OOM as model-weight OOM without separating pools/reservations.
- Capturing dynamic request shapes with no admission/bucket policy.
- Benchmarking only replays and omitting capture, input copy, and graph-cache growth.
- Switching allocator flags and model/compile settings simultaneously.

## Mastery gate

Build a graph cache for two justified stable shapes that preserves output correctness, obeys a VRAM budget, and falls back safely for others. Explain its allocator/pool behavior and demonstrate why it improves a measured launch-bound workload but not a compute-bound one.

## Primary sources, code paths, and OSS bridge

- [`torch.cuda.CUDAGraph` API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)
- [PyTorch CUDA semantics and memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [CUDA Graph Trees documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
- [NVIDIA CUDA Graphs programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [vLLM CUDA graph documentation](https://docs.vllm.ai/en/latest/design/cuda_graphs/)

Read `torch/cuda/graphs.py`, `torch/_inductor/cudagraph_trees.py`, allocator code under `c10/cuda/`, and vLLM graph/cache management at your pinned revision. Good contributions are output-liveness tests, capture eligibility diagnostics, memory regressions with snapshots, or shape-cache accounting—not broad graph enablement toggles.

## Next

Module 42 gives you a disciplined method for reducing, diagnosing, and upstreaming compiler/runtime regressions.
