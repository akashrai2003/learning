# Module 03 — Python Performance and Data Model

> **Phase:** I — Engineering foundations  
> **Time:** Week 2; three focused 3-hour sessions  
> **Prerequisites:** Modules 01–02 and practical Python

## Existing inference curriculum bridge

Connect this module to [Module 01 — `inference-system-map.md`](../../inference-engineering-curriculum/modules/01-inference-system-map.md), [Module 05 — `performance-modeling-and-roofline.md`](../../inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md), [Module 34 — `continuous-batching-chunked-prefill-and-prefix-cache.md`](../../inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md), and [Module 44 — `profiling-benchmarking-and-correctness.md`](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). GPU kernels can be excellent while Python scheduling, tokenization, serialization, or synchronization still owns user latency.

## Why this matters and outcomes

PyTorch deliberately gives Python a dynamic front end. That freedom makes ergonomic model code, but it also makes mutable defaults, iterator exhaustion, import-time side effects, implicit tensor conversions, and per-token Python loops easy to ship. Learn enough of CPython’s execution and object model to make code predictable before asking a compiler to capture it.

You will be able to select value objects versus mutable state holders, identify the GIL’s actual implications, profile CPU control-plane work correctly, and replace Python-over-tensor loops with vectorized tensor work only when that preserves semantics.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P03 — Python performance and data model**.

1. [Python data model](https://docs.python.org/3/reference/datamodel.html) — learn the API or system contract.
2. [Python profiling](https://docs.python.org/3/library/profile.html) — study the underlying design, paper, or specification.
3. [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [Python data model](https://docs.python.org/3/reference/datamodel.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Python profiling](https://docs.python.org/3/library/profile.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: a serving process has a control plane and a tensor plane

The tensor plane should send coarse, vectorized work to PyTorch/ATen/CUDA. The control plane admits requests, validates schemas, looks up caches, forms batches, handles cancellation, streams outputs, and logs metrics. Python is appropriate for complex policy and orchestration; it is inappropriate for a million scalar operations or a synchronization per generated token.

CPU time is not one thing:

\[
t_{\mathrm{request}}=t_{\mathrm{queue}}+t_{\mathrm{Python}}+t_{\mathrm{CPU\ native}}+
t_{\mathrm{GPU}}+t_{\mathrm{sync}}+t_{\mathrm{network}}.
\]

The host may enqueue GPU kernels quickly while device work continues asynchronously. A Python timing loop that calls `.item()`, accesses a CUDA result, formats a tensor, or invokes `torch.cuda.synchronize()` can insert \(t_{\mathrm{sync}}\) and falsely attribute it to a harmless line.

## Data-model and performance theory

An instance lookup generally resolves data descriptors, instance dictionary, non-data descriptors/class attributes, then `__getattr__`. `@property`, `__getattribute__`, `__iter__`, `__enter__`, `__call__`, and context-manager methods are executable behavior—not decoration. Use `dataclass(frozen=True)` for immutable configuration/value objects, `Enum` for closed modes, and explicit `Protocol`/ABC boundaries only where substitution is truly needed. Do not make a giant inheritance tree around `nn.Module` before the contract is stable.

Mutable defaults are evaluated once at function definition, so use `field(default_factory=list)` or `None` then create a list. A generator is single-pass; materializing it may cost memory but reusing it without realizing it is exhausted costs correctness. Iterators, lazy imports, and context managers are particularly relevant for streaming token sources and resources that must be closed when a request cancels.

CPython’s GIL permits one thread to execute Python bytecode at a time in the traditional build, while native extensions may release it. Threads can help overlap I/O or native work; they do not make a pure-Python request scheduler CPU-parallel. Multiprocessing adds isolation and true CPU parallelism but forces serialization, process lifecycle, and CUDA-safety decisions. `asyncio` is cooperative concurrency: an accidental blocking tokenizer call blocks the event loop. Measure before choosing processes, threads, async tasks, or a native/vectorized implementation.

For tensor code, a Python `for` loop launches or dispatches repeatedly, whereas one operation such as `x.square().sum(dim=-1)` lets PyTorch select fused/native kernels. Do not assume vectorization always wins: a huge broadcast can materialize or drive more memory traffic than a compact loop. Establish a shape-distributed benchmark and inspect allocations.

## Implementation ladder in the proposed `torch-lab/`

1. **Reference:** implement `Request`, `GenerationConfig`, and `RunRecord` dataclasses in `src/torch_lab/config.py`; use a simple FIFO Python scheduler over synthetic request objects. Include explicit lifecycle states: `QUEUED`, `RUNNING`, `FINISHED`, `CANCELLED`.
2. **Optimized:** replace a scalar token-score loop with an equivalent vectorized PyTorch operation; create `benchmarks/python_control_plane.py` using `time.perf_counter_ns`, warm-up, and no CUDA synchronization unless it is the measured boundary. Profile with `cProfile` and verify the same outputs/state transitions.
3. **Integrated:** add an async streaming adapter whose policy logic is testable without HTTP, bounded queues for backpressure, cancellation cleanup, and tracing fields that connect queue time to GPU timing introduced later.

## Three focused sessions

1. **Session A — language contracts (3h):** write the state/value dataclasses, inspect `repr`, equality, hashing, and default behavior, then test invalid transitions. Add type checking as a development tool but keep runtime validation at the external boundary.
2. **Session B — measure the control plane (3h):** compare a Python loop, vectorized tensor form, and deliberately over-broadcast form across representative sizes. Use `dis.dis` on a tiny loop and `cProfile` on the scheduler. Explain the result in terms of launches, allocation, and complexity rather than intuition.
3. **Session C — concurrency (3h):** build a fake asynchronous tokenizer and producer/consumer queue; cancel requests midstream; ensure no task leaks or duplicate finalization. Repeat with a blocking function to observe event-loop starvation.

## Experiments and deliverables

Publish a memo with flamegraph/cProfile output, raw timing samples, batch/token distributions, and a hypothesis for the dominant stage. Test that a cancellation releases scheduler state, that `GenerationConfig` is immutable, and that a vectorized rewrite preserves exact results for random edge shapes. Add a lint/type/test command to CI, but do not claim static types prove tensor shapes.

## Common failure modes

- Optimizing list comprehensions while a hidden CUDA synchronization dominates elapsed time.
- Sharing mutable request or `dict` defaults across users, producing cross-request state corruption.
- Using threads for CPU-bound Python hoping to bypass the GIL, then masking the real queueing bottleneck.
- Creating one tensor or kernel call per token/head instead of expressing a tensor batch.
- Adding `async` to a function whose body still does blocking filesystem, subprocess, or CPU work.

## Mastery gate

Given a trace with high TTFT, separate queue, Python, native CPU, GPU, and synchronization time with evidence. Refactor a scalar reference into a tensor implementation, prove parity, and show a benchmark where it wins at a declared shape. Demonstrate cancellation-safe scheduler behavior under 100 synthetic concurrent requests.

## Primary sources and canonical code paths

- [Python data model](https://docs.python.org/3/reference/datamodel.html), [asyncio](https://docs.python.org/3/library/asyncio.html), [threading/GIL](https://docs.python.org/3/library/threading.html), [profiling](https://docs.python.org/3/library/profile.html), and [disassembly](https://docs.python.org/3/library/dis.html).
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) and [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html).
- Inspect vLLM’s [scheduler package](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core) as a production control-plane example and PyTorch [`torch/utils/data`](https://github.com/pytorch/pytorch/tree/main/torch/utils/data) for process/worker boundaries.

## OSS bridge and next

Small, typed bug reproductions and lifecycle tests are valuable contributions to serving projects as well as PyTorch. Do not optimize Python blindly: use the same evidence standard later applied to kernels. Next, give every tensor numerical semantics in [Module 04](04-tensor-math-and-numerical-contracts.md).
