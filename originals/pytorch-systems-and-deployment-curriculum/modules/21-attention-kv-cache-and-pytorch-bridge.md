# Module 21 — Attention, KV Cache, and the PyTorch Bridge

> **Phase:** III — Model architecture and implementation  
> **Time:** 12–15 focused hours across four sessions  
> **Prerequisites:** Modules 08–20 and the tested decoder from Module 20

## Existing inference curriculum bridge

This module directly implements the contracts behind [Module 06 — Transformer Forward Pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), [Module 10 — MHA, Masking, and Causal Attention](../../inference-engineering-curriculum/modules/10-mha-masking-and-causal-attention.md), [Module 12 — MQA, GQA, and KV Cache](../../inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md), [Module 29 — Attention IO Complexity and Online Softmax](../../inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md), [Module 30 — FlashAttention Evolution](../../inference-engineering-curriculum/modules/30-flashattention-evolution.md), and [Module 31 — PagedAttention and Memory Management](../../inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md). Keep the PyTorch implementation as a correctness bridge while you learn production cache allocators and kernels.

## Why this matters and outcomes

Autoregressive decoding turns a Transformer forward into an inference system. Recomputing old K/V projections and attention at every generated token wastes work. A KV cache removes that repetition but creates state ownership, layout, masking, allocation, batching, reorder, and memory-capacity problems.

By the end, you can implement and test full-sequence versus cached decode parity; quantify prefill/decode compute and cache bytes; design a cache contract with offsets and reset/reorder behavior; handle MHA/GQA/MQA head mappings; and explain which parts PyTorch SDPA solves versus which vLLM-style paged serving must solve.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P21 — Attention, KV cache, and PyTorch bridge**.

1. [SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — learn the API or system contract.
2. [MQA paper](https://arxiv.org/abs/1911.02150) — study the underlying design, paper, or specification.
3. [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch Transformer tutorial](https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model, equations, and shapes

For L layers, B sequences, H query heads, Hkv key/value heads, token capacity C, and head width Dh, store each layer’s K and V as [B,Hkv,C,Dh]. With element size e bytes, raw cache capacity is approximately:

2 * L * B * Hkv * C * Dh * e bytes.

During prefill, a prompt of T tokens creates Q [B,H,T,Dh], K/V [B,Hkv,T,Dh], and attention reads the whole prefix. During one-token decode at absolute position t, new Q is [B,H,1,Dh], new K/V are [B,Hkv,1,Dh], and attention reads cached K/V through t, producing scores [B,H,1,t+1]. This avoids recomputing old projections, yet decode remains bandwidth/latency sensitive because each new query reads a growing cache.

A reference cache can append with concatenation for clarity; it is intentionally allocation-heavy. A real fixed-capacity cache preallocates K/V and writes into positions [start:start+q_len]. Its contract must state layer, batch-slot mapping, absolute start, valid length, capacity, dtype/device, mutation ownership, and reset/reorder semantics. Cache state is inference state: create/use it under no_grad or inference_mode unless you intentionally implement differentiable cache behavior. Never accidentally mutate it across unrelated requests.

Grouped-query attention has H query heads mapped to fewer Hkv KV heads. If r=H/Hkv, query head h maps to KV head floor(h/r) for divisible groups. A naive repeat_interleave can materialize expanded K/V and conceal the intended memory benefit. PyTorch SDPA exposes enable_gqa, but its exact backend and layout behavior must be profiled and tested on the installed version; do not assume it implements your serving cache layout.

Masking is subtle for cached chunks. Full causal attention masks key index k greater than absolute query position q. A one-token query at position t should see keys 0 through t, not merely a generic rectangular “upper-left” causal pattern. For non-square query/key shapes, construct or use an offset-aware causal bias and test it. Do not toggle is_causal blindly and assume it has the cache offset you need.

PagedAttention maps logical token blocks to non-contiguous physical pages so sequences can share, grow, and release KV capacity efficiently. PyTorch concatenation/preallocation proves semantic cache correctness; it does not provide continuous batching, block tables, eviction, prefix sharing, or page-aware kernels.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Extend torch-lab/models/decoder.py with a per-layer cache that appends K/V and a generate loop. Compare every cached-token logit with full-prefix decoding for small prompts and generated continuations.
2. Test cache shape, valid length, reset, capacity error, dtype/device, and future-token masking with asymmetric B/H/Hkv/T/Dh cases.

**Optimized**

3. Replace concatenation with preallocated K/V buffers and explicit write offsets. Benchmark allocation count, peak memory, prefill, and decode while preserving full-cache parity.
4. Add MHA, MQA, and GQA configurations. Verify head mapping without materialized K/V expansion where the implementation contract says it should avoid it.

**Integrated**

5. Add cache clone/reorder operations for a toy beam-search batch, request-slot reset, chunked prefill, and an offset-aware mask test. Keep batch-slot ownership explicit.
6. Profile the same model with PyTorch SDPA and record backend/layout/mask behavior. Compare it with the semantics needed by paged serving; do not claim a kernel or scheduler equivalence.

## Four focused sessions

1. **Full versus incremental math (3h):** derive cache bytes/work, implement append-cache decode, and make token-by-token logits match the full oracle.
2. **Cache representation (3h):** preallocate, write offsets, reset/reorder, test capacity, and explain mutation/lifetime rules.
3. **Heads and masks (3h):** implement MHA/GQA/MQA mappings and cached rectangular attention tests; expose one wrong mask that still produces plausible text.
4. **Systems bridge (3h):** profile prefill/decode, study page/block-table concepts, and map every reference-cache operation to a production runtime responsibility.

## Experiments and deliverables

Commit cache.py, full-versus-cached parity tests, cache-byte calculator, MHA/GQA/MQA matrix, offset-mask test, allocation/latency report, and a page-table design note. Report L/B/H/Hkv/C/Dh/dtype, allocated versus active cache bytes, prefill/decode token rates, cache mutation policy, and parity tolerance.

## Common failure modes

- Comparing only the final generated string instead of all cached logits.
- Repeated torch.cat growth hidden inside a benchmark.
- Applying a non-offset causal mask to cached rectangular attention.
- Forgetting cache reset/reorder when batch slots or beams change.
- Expanding GQA K/V tensors and losing the intended memory advantage.
- Treating PyTorch SDPA as a paged allocator, scheduler, or universal backend guarantee.
- Using mutable inference cache in a gradient-tracked training path.

## Mastery gate

Produce a cache whose cached logits match full-prefix logits across chunk boundaries, varied shapes, MHA/GQA/MQA, and beam reorder/reset tests. Given a model config, calculate cache capacity and identify whether decode is projection-, attention-, or cache-read dominated. Explain exactly what a paged engine adds beyond the reference.

## Primary sources and canonical code paths

- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch attention bias API](https://docs.pytorch.org/docs/stable/nn.attention.bias.html)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [vLLM documentation](https://docs.vllm.ai/)
- Inspect [functional.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py), [torch/nn/attention](https://github.com/pytorch/pytorch/tree/main/torch/nn/attention), and vLLM’s attention/cache source after completing the reference.

## OSS bridge and next

A strong inference contribution begins with full-versus-cached parity, explicit cache layout, and a minimised mask or slot-lifecycle failure. The next curriculum phase can safely move into torch.compile, export, quantization, native kernels, and serving runtimes because you now own the model/cache oracle.
