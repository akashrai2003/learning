# Module 14 — Long Context and Memory

> **Phase:** 3 — Advanced Architectures  
> **Time:** Week 7; two 3-hour sessions plus a 2–4 hour systems lab and mastery gate  
> **Prerequisites:** Modules 10–12; RoPE scaling, KV-cache arithmetic, GPU memory hierarchy, and prefill/decode benchmarking

## Why this matters

“Supports 128K” combines several independent claims: the position scheme accepts those indices, kernels can execute the workload, memory can hold it at useful concurrency, the scheduler remains responsive, and the model can actually use distant information. Long-context inference stresses attention I/O, KV capacity, temporary workspaces, fragmentation, and time to first token (TTFT). Paged allocation, prefix caching, and chunked prefill improve different parts of this problem; none makes context free.

## Outcomes

You will be able to:

- build a complete weight/KV/workspace memory budget;
- explain FlashAttention, PagedAttention, prefix caching, sliding windows, and chunked prefill without conflating them;
- implement a block allocator and logical-to-physical cache table;
- quantify throughput–latency–fragmentation trade-offs; and
- evaluate effective context with more than a single needle retrieval.

## Mental model

Treat GPU memory like a small warehouse. Model weights are permanent shelving. Every live request grows a stack of KV records. A contiguous allocator reserves a worst-case aisle for each request and leaves holes as requests finish. Paging cuts records into fixed-size blocks and gives each request a block table, allowing non-contiguous physical storage. Prefix caching lets requests point to identical immutable blocks. The attention kernel follows the table to read the logically contiguous history.

## Theory, math, and shapes

For request lengths $S_i$, full-attention KV bytes are

$$
2N_\text{layers}H_{kv}D\,b\sum_i S_i.
$$

This is the payload floor; allocator metadata, scales, temporary buffers, CUDA graphs, and weights require additional space. GQA reduces $H_{kv}$. FP8 cache reduces $b$ but introduces scale and accuracy concerns. PagedAttention reduces fragmentation and enables sharing; it does **not** reduce payload bytes per unique token.

With block size `C`, physical K/V may be `[num_blocks,C,Hkv,D]`, while request `r` owns `block_table[r]: [num_logical_blocks]`. A token's logical block is `t//C` and in-block offset is `t%C`. Only the final block can be partially filled, bounding internal waste per request to fewer than `C` token slots.

Naive prefill materializes score/probability tensors with quadratic auxiliary memory. FlashAttention remains exact and retains quadratic arithmetic while using tiled, online softmax to reduce HBM traffic and auxiliary storage. Decode attends one or a few queries to a growing history, so its work and KV reads grow roughly linearly with context.

Local or sliding-window attention changes the model's semantics: a local layer retains only the most recent `W` positions and bounds its cache term by `min(S,W)`. If a hybrid has `Lf` full layers and `Lw` local layers, its approximate cache is `2*Hkv*D*b*(Lf*S + Lw*min(S,W))`. Periodic global layers preserve some long-range paths while local layers cap most state. Paging can manage either layout, but it cannot turn a full-attention checkpoint into a local one without changing outputs.

Prefix caching hashes each full block together with its prefix identity, token IDs, and correctness-relevant extras such as adapter or multimodal identity. A hit skips repeated prefill computation but still consumes referenced cache space. Chunked prefill splits a long prompt across scheduler iterations, letting decode requests run between chunks. It can improve inter-token latency and fairness, but may increase one request's TTFT and does not reduce its final KV footprint.

## Implementation ladder

1. Add a CPU-simulated `BlockPool` in `inference-lab/engine/block_pool.py`: fixed block IDs, free queue, reference counts, allocate, append, free, and invariant checks.
2. Add `engine/paged_kv_cache.py` with physical K/V tensors, per-request block tables, slot mapping, and a materialize-for-reference function.
3. Adapt `kernels/attention.py` to consume block metadata. First gather into a contiguous reference; later replace it with direct paged reads.
4. Add full-block prefix hashes and copy-on-reference semantics in `engine/prefix_cache.py`. Include a namespace/salt hook.
5. Extend `engine/scheduler.py` with a token budget and chunked-prefill quantum. Keep admission, allocation, and compute decisions separately testable.

## Two 3-hour sessions

**Session A — paging and correctness**

- 0:00–0:45: calculate a realistic memory budget and concurrency ceiling.
- 0:45–1:30: implement and property-test the block pool.
- 1:30–2:20: implement paged append, lookup, and reference materialization.
- 2:20–3:00: compare logits from contiguous and paged cache paths.

**Session B — reuse and scheduling**

- 0:00–0:45: add prefix hashing, reference counts, and eviction.
- 0:45–1:30: implement chunked prefill with a per-step token budget.
- 1:30–2:15: run mixed short-decode/long-prefill workloads.
- 2:15–3:00: evaluate long-context correctness and interpret tail latency.

## Experiments and deliverables

Produce:

- `tests/test_block_pool.py` with randomized allocate/append/free traces and invariant checks;
- `tests/test_paged_cache.py` proving contiguous-versus-paged logits and safe shared-prefix lifetimes;
- `benchmarks/bench_long_context.py` reporting TTFT, inter-token latency, throughput, peak memory, preemptions, and prefix-hit rate;
- a block-size sweep measuring internal waste and metadata/kernel effects;
- a full-versus-sliding-versus-periodic-global table of cache bytes, latency, and distant-token behavior;
- a repeated-system-prompt experiment showing cold and warm TTFT; and
- `notes/module14.md` with predicted versus measured memory at several lengths.

Quality evaluation must vary evidence position and length. Include multi-needle, aggregation, or multi-hop cases inspired by RULER, plus a short-context regression. Record tokenizer-derived lengths, not character counts.

## Common failure modes

- Equating configured maximum length with effective model context.
- Forgetting that requested output tokens also need cache capacity.
- Claiming paging compresses KV data rather than reducing waste and enabling sharing.
- Reusing a prefix across different model weights, adapters, RoPE settings, media, or trust domains.
- Mutating a shared prefix block instead of allocating a new tail.
- Evicting a block with a nonzero reference count.
- Allowing a huge prefill to starve active decodes, or choosing chunks so small that overhead dominates.
- Ignoring temporary allocations and graph-capture pools in the memory budget.
- Comparing throughput without arrival pattern and latency constraints.
- Using only a trivial needle test and overlooking short-context regressions.

## Mastery gate

Given a model configuration, VRAM budget, block size, dtype, and five request lengths, predict whether allocation succeeds and bound internal fragmentation. Your randomized block-pool tests must finish with every reference count and free-list invariant intact. Contiguous and paged logits must match, shared prefixes must survive one consumer finishing, and your scheduler report must explain a measured TTFT/inter-token-latency trade-off.

## Primary sources

- Kwon et al., [“Efficient Memory Management for Large Language Model Serving with PagedAttention”](https://arxiv.org/abs/2309.06180).
- vLLM, [PagedAttention design](https://docs.vllm.ai/en/latest/design/paged_attention/) and [automatic prefix-caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/).
- Agrawal et al., [“SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills”](https://arxiv.org/abs/2308.16369).
- Hsieh et al., [“RULER: What’s the Real Context Size of Your Long-Context Language Models?”](https://arxiv.org/abs/2404.06654).
- Xiao et al., [“Efficient Streaming Language Models with Attention Sinks”](https://arxiv.org/abs/2309.17453).

## Code paths to read

Your paths are `engine/block_pool.py`, `engine/paged_kv_cache.py`, `engine/prefix_cache.py`, `engine/scheduler.py`, `tests/test_block_pool.py`, `tests/test_paged_cache.py`, and `benchmarks/bench_long_context.py`.

## OSS bridge

Trace vLLM’s [`block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py), [`single_type_kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py), and [optimization guidance](https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md). Identify which component owns memory, prefix identity, and scheduling.

## Next

Module 15 introduces variable-sized non-text inputs whose preprocessing, token expansion, and cache identity must join this same engine contract.
