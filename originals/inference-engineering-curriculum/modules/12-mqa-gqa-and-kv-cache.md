# Module 12 — MQA, GQA, and the KV Cache

> **Phase:** 2 — Architecture Core  
> **Time:** Week 6; two 3-hour sessions plus a 2–4 hour independent lab and mastery gate  
> **Prerequisites:** Modules 03–11; causal attention, RoPE position offsets, tensor views, and memory-footprint arithmetic

## Why this matters

Autoregressive generation would be wasteful if every new token recomputed keys and values for the entire prefix. A KV cache retains those projections, reducing repeated work, but its size grows with context and concurrency. Multi-query attention (MQA) and grouped-query attention (GQA) reduce that cache and the bytes read on every decode step by sharing K/V heads. They are architectural properties of a checkpoint, not serving switches.

## Outcomes

You will be able to:

- distinguish MHA, MQA, and GQA from their projection and cache shapes;
- calculate per-request and fleet-level KV-cache memory;
- implement contiguous cached prefill, append, and one-token/chunk decode;
- map query heads to KV heads without physically duplicating the cache; and
- prove cached generation is equivalent to a full causal forward pass.

## Mental model

Queries are temporary readers; keys and values are the growing memory they read. MHA gives every query head a private memory stream. MQA makes all query heads share one K/V stream. GQA partitions query heads into groups, each with one shared stream—a compromise between representation capacity and memory bandwidth. The cache is model state belonging to a particular sequence, layer, position history, dtype, and model version.

## Theory, math, and shapes

Let query-head count be $H_q$, KV-head count $H_{kv}$, head dimension $D$, and $g=H_q/H_{kv}$. The model requires `Hq % Hkv == 0`.

- Q projection output: `[B,T,Hq,D]`
- K and V projection outputs: `[B,T,Hkv,D]`
- per-layer cache, common logical layout: `[B,Hkv,S,D]` for K and V
- attention scores after grouping: `[B,Hq,L,S]`

MHA has $H_{kv}=H_q$, MQA has $H_{kv}=1$, and GQA lies between. Query head `h` uses KV head `floor(h/g)` when groups are contiguous. A reference may use `repeat_interleave` to expand K/V to `Hq`; an inference kernel should map heads logically and read the compact cache.

For one sequence, a full-attention cache consumes approximately

$$
\text{bytes}=2\,N_\text{layers}\,S\,H_{kv}\,D\,b,
$$

where `2` means K plus V and $b$ is bytes per element. Multiply by active sequence lengths, not simply maximum batch size, when estimating a dynamic server. Cache quantization changes $b$ but may add scale metadata and accuracy constraints.

Prefill computes and stores K/V for all prompt tokens. Decode projects only `L` new tokens, appends them, and attends to total length `S_old+L`. RoPE uses absolute new positions; the causal boundary is `S_old+i`. Cache use removes repeated projections, but each layer still reads prior K/V, so long-context decode is often memory-bandwidth-bound.

## Implementation ladder

1. Add `inference-lab/engine/kv_cache.py` with one fixed-capacity contiguous K/V tensor per layer, a current length, bounds checks, `append`, `view`, `reset`, and batch reorder.
2. Extend `reference/attention.py` with explicit `num_q_heads` and `num_kv_heads`. First validate a materialized `repeat_kv` reference.
3. Add a grouped path in `kernels/attention.py` that maps query heads to compact K/V without persisting repeated tensors.
4. Implement `prefill()` and `decode_step()` in `engine/generate.py`. Make cache mutation explicit and return the new logical length.
5. Test one token at a time, irregular chunks, and two independent sequences whose states must never alias.

This cache is intentionally contiguous. Module 14 replaces its allocation policy with blocks while preserving attention semantics.

## Two 3-hour sessions

**Session A — head sharing**

- 0:00–0:40: derive MHA/MQA/GQA projection and cache shapes.
- 0:40–1:20: build a memory table across `Hkv`, context, and dtype.
- 1:20–2:15: implement `repeat_kv` reference and compact grouped attention.
- 2:15–3:00: test head mapping, invalid configs, and parity.

**Session B — stateful decode**

- 0:00–0:40: specify cache ownership, lifecycle, and position invariants.
- 0:40–1:35: implement prefill, append, and decode.
- 1:35–2:20: prove cached/full equivalence over irregular chunkings.
- 2:20–3:00: benchmark latency and bytes read as context grows.

## Experiments and deliverables

Produce:

- `tests/test_gqa.py` comparing compact GQA with explicitly repeated K/V for `Hkv={1,2,Hq}`;
- `tests/test_kv_cache.py` covering capacity, reset, reorder, sequence isolation, and full-versus-cached logits;
- `benchmarks/bench_decode.py` sweeping context length and reporting per-token latency plus tokens/s;
- `notes/module12.md` with predicted and measured cache bytes for one model configuration; and
- an experiment proving that materialized K/V repetition raises allocated memory without changing output.

Report prefill throughput separately from decode throughput. Include batch size and total live tokens; “tokens/s” without workload shape is not comparable.

## Common failure modes

- Treating `num_kv_heads` as a runtime optimization and loading an MHA checkpoint into an MQA layout.
- Appending before computing the correct position or attending to a token twice.
- Using a square causal mask for `L` new queries against `S_old+L` keys.
- Repeating K/V into a permanent cache, erasing GQA's memory benefit.
- Sharing cache storage between requests, beams, or model replicas.
- Forgetting to reorder cache rows during beam search.
- Growing tensors with `torch.cat` every step, causing repeated allocation and copies.
- Ignoring padding and per-sequence lengths in a ragged batch.
- Quantizing cache values without checkpoint/backend-compatible scales.
- Comparing sampled text rather than deterministic logits when checking equivalence.

## Mastery gate

For `32` query heads, `8` KV heads, `D=128`, `32` layers, `S=16384`, and fp16, derive the per-sequence cache size and the MHA/MQA alternatives. Then demonstrate that full forward, one-token decode, and at least three irregular chunkings produce equivalent logits under a justified tolerance. Peak allocated memory for your compact path must remain below the explicit-repeat reference.

## Primary sources

- Shazeer, [“Fast Transformer Decoding: One Write-Head is All You Need”](https://arxiv.org/abs/1911.02150).
- Ainslie et al., [“GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”](https://aclanthology.org/2023.emnlp-main.298/).
- Hugging Face Transformers, [cache strategies](https://huggingface.co/docs/transformers/kv_cache) and [`cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py).
- Hugging Face Transformers, [Llama GQA `repeat_kv` and attention implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

## Code paths to read

Your paths are `engine/kv_cache.py`, `reference/attention.py`, `kernels/attention.py`, `engine/generate.py`, `tests/test_gqa.py`, `tests/test_kv_cache.py`, and `benchmarks/bench_decode.py`.

## OSS bridge

Compare dynamic and static cache trade-offs in Transformers. In vLLM, preview [`single_type_kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py), but defer paging and prefix reuse to Module 14.

## Next

Module 13 replaces one dense FFN with routed experts and confronts conditional computation, dispatch, and expert-weight traffic.
