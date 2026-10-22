# Module 31 — PagedAttention and KV Memory Management

**Phase:** 5 — Attention and decoding systems  
**Pacing:** Week 16, paired with Module 32; three 3-hour sessions plus a weekend cache simulator  
**Prerequisites:** Modules 12, 27, 29–30; KV-cache shapes, quantized caches, decode attention, and basic allocator concepts

## Why this matters

Request lengths are unknown and change one token at a time. Reserving a contiguous maximum-length KV tensor wastes memory; reallocating growing tensors fragments memory and copies state. PagedAttention separates each request’s logical token order from fixed-size physical KV blocks, allowing on-demand allocation, continuous batching, sharing, and copy-on-write.

## Outcomes

You can:

- derive logical-to-physical KV address translation;
- calculate block bytes, metadata, and internal fragmentation;
- implement allocation, append, free, sharing, and copy-on-write;
- reason about block size, prefix reuse, eviction, and preemption;
- verify paged-attention output against a contiguous oracle.

## Mental model: block table as a page table

Let each physical KV block hold \(P\) tokens. For request \(r\) and logical token \(t\),

\[
b_\text{logical}=\lfloor t/P\rfloor,\qquad
o=t\bmod P,
\]

\[
b_\text{physical}=\text{block\_table}[r,b_\text{logical}].
\]

The attention kernel reads K/V at physical block \(b_\text{physical}\), offset \(o\), and the appropriate layer/head/channel. Conceptually a uniform block costs

\[
\text{block bytes}=P\cdot2LH_{kv}D\cdot\text{bytes(dtype)}
\]

plus scales/metadata. Real engines often allocate separate per-layer or per-cache-group pools; hybrid windows and heterogeneous head sizes cannot always share one block geometry.

This “block” is neither an operating-system page nor a CUDA thread block. It is the runtime’s KV allocation unit.

## Why paging helps

Contiguous reservation for \([B_\max,S_\max]\) wastes unused tail capacity. Paged allocation requests a new block only when the current block fills. Internal fragmentation is at most \(P-1\) token slots per live sequence per cache group, rather than an entire unused maximum length.

Paging enables:

- noncontiguous growth without copying prior K/V;
- immediate block recycling when a request finishes;
- shared full-prefix blocks with reference counts;
- copy-on-write when beams or parallel samples diverge;
- swapping/offload or recomputation under pressure.

It does not make memory infinite. Block tables, free-list contention, prefix indexes, quantization scales, graph buffers, and allocator reserve all consume memory.

## Block-size trade-offs

Larger \(P\) reduces block-table entries and may improve memory coalescing/kernel efficiency, but increases tail fragmentation and makes exact prefix matches coarser. Smaller \(P\) improves allocation granularity and reuse but raises metadata/addressing overhead.

Measure with workload distributions, not one sequence. Useful metrics:

\[
\text{utilization}=
\frac{\text{live logical token slots}}
{\text{allocated physical token slots}},
\qquad
\text{fragmentation}=1-\text{utilization}.
\]

Also track allocation failures, evictions, preemptions, recomputation tokens, swap bytes, prefix-hit tokens, and cache-block churn.

## Prefix sharing and correctness

A reusable block identity must include the complete token prefix dependency and any state that affects KV: model/adapters, modality inputs, cache dtype/scales, position scheme, tenant/cache salt, and possibly request-specific context. Hash collisions need verification or a sufficiently strong hash policy.

Only fully computed blocks are straightforward to share. A partially filled shared block needs copying before mutation. Reference counts must survive request cancellation and errors; double-free or premature reuse causes cross-request corruption.

## Implementation ladder

### 1. Reference

Create `inference-lab/engine/cache/paged_cache.py` with a deterministic free list, per-request block tables, refcounts, append, fork, copy-on-write, and free. Store tiny NumPy/PyTorch K/V blocks and expose invariant checks.

### 2. Optimized

Add batched block-table preparation and a paged attention gather/online-softmax implementation. Avoid concatenating a contiguous cache. Introduce quantized scale sidecars and multiple cache groups.

### 3. Integrated

Build a discrete-event workload simulator with arrivals, input/output length distributions, prefix families, and finite block pools. Then compare its capacity/fragmentation predictions with one real engine’s cache logs and serving metrics.

## Three 3-hour sessions

1. **Allocator:** derive block bytes and implement allocate/append/free with invariants and randomized state-machine tests.
2. **Sharing/attention:** add fork/refcount/copy-on-write and verify paged versus contiguous attention token by token.
3. **Workload study:** sweep page size, prefix reuse, concurrency, and memory pressure; inspect production block manager and attention addressing.

The weekend simulator reports TTFT/TPOT or service time, cache utilization, fragmentation, prefix-hit rate, preemption, and fairness for a pinned workload.

## Experiments and deliverables

Commit:

- `engine/cache/paged_cache.py`;
- `engine/cache/workload_sim.py`;
- `tests/test_paged_cache_state_machine.py`;
- `benchmarks/results/paged-cache-block-sweep.json`;
- `notes/source_walks/pagedattention-allocation.md`.

Use fixed arrival/length seeds and retain raw traces. Include cancellations, OOM admission, shared-prefix divergence, last partial blocks, and hybrid cache groups.

## Common failure modes

- Confusing logical blocks, physical blocks, and CUDA blocks.
- Calculating payload without K and V, layers, scale metadata, or cache groups.
- Optimizing average fragmentation while causing allocation tail failures.
- Mutating a shared partial block without copy-on-write.
- Reusing a prefix across incompatible adapters, modalities, or tenants.
- Defragmenting by copying entire live caches and hiding the pause.
- Validating allocator counts without comparing attention outputs.

## Mastery gate

Implement a randomized paged-cache state machine that never leaks, double-frees, or aliases mutable blocks; derive its capacity/fragmentation; and prove paged attention matches a contiguous oracle across forks, frees, and reuse.

## Canonical code paths and OSS bridge

Read the [PagedAttention paper](https://arxiv.org/abs/2309.06180), vLLM’s [paged-attention kernel design](https://docs.vllm.ai/en/latest/design/paged_attention/) and prefix-cache implementation, then follow `vllm/v1/core/kv_cache_manager.py`, block-pool/cache utilities, block-table preparation, and backend kernel. Compare TensorRT-LLM’s [KV cache system](https://nvidia.github.io/TensorRT-LLM/1.2.0/features/kvcache.html). Good OSS work includes allocator invariant tests, cache-event tooling, hybrid-group accounting, and cancellation/reuse regressions.

## Primary and authoritative sources

- [Efficient Memory Management with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM repository](https://github.com/vllm-project/vllm)
- [vLLM paged-attention design](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [TensorRT-LLM KV cache system](https://nvidia.github.io/TensorRT-LLM/1.2.0/features/kvcache.html)

## Next

Module 32 completes the token step after logits: deterministic processing, truncation, random sampling, stop semantics, and efficient vocabulary-wide kernels.
