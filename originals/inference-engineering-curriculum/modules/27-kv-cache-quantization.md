# Module 27 — KV-Cache Quantization

**Phase:** 4 — Quantization  
**Pacing:** Week 14, paired with Module 28; three 3-hour sessions plus a weekend long-context study  
**Prerequisites:** Modules 12, 19, 25–26; cached attention, GQA, calibration, FP8/block formats, and quality evaluation

## Why this matters

The KV cache grows with every request token and is read repeatedly during decode. At long context or high concurrency it can dominate capacity and bandwidth even after weights are quantized. KV quantization can extend context and batch size, but keys and values have different distributions, scale axes affect attention error, and an unfused dequant path can erase throughput gains.

## Outcomes

You can:

- derive KV-cache memory for MHA/GQA and hybrid attention;
- choose and implement scale granularity for keys and values;
- compare FP8 and lower-bit research schemes;
- evaluate long-context correctness, not only short PPL;
- integrate quantized KV with paged prefill/append/decode kernels.

## Mental model: a growing, repeatedly consumed activation

For batch \(B\), cached length \(S\), \(L\) uniform layers, \(H_{kv}\) KV heads, head dimension \(D\), and \(b\) bits/element,

\[
\text{KV bytes}=2BLSH_{kv}D\frac{b}{8}+\text{scale/metadata bytes}.
\]

GQA reduces \(H_{kv}\); sliding-window or hybrid models require a sum over layer groups and retained lengths rather than one \(S\).

At decode step \(t\), query \(Q_t[B,H_q,D]\) consumes cached
\(K,V[B,t,H_{kv},D]\):

\[
P_t=\operatorname{softmax}(Q_tK^\top/\sqrt D),\qquad O_t=P_tV.
\]

Key error perturbs logits before an exponential; value error perturbs the weighted sum. Therefore identical quantizers need not suit K and V.

KIVI’s distribution study motivates per-channel key quantization and per-token value quantization at very low bit widths. KVQuant additionally studies pre-RoPE key quantization, nonuniform datatypes, and sparse outlier handling. Production FP8 caches often use coarser per-tensor or per-layer scales because kernels and metadata are simpler. Treat paper algorithms and supported runtime formats separately.

## Scale and layout choices

For affine quantization:

\[
q=\operatorname{clamp}(\operatorname{round}(x/s)+z),\qquad
\hat x=s(q-z).
\]

Possible scale shapes include:

- per layer/tensor: scalar `k_scale`, `v_scale`;
- per head: \([L,H_{kv}]\);
- per token/vector: \([B,S,H_{kv},1]\);
- per channel across tokens: \([L,H_{kv},D]\);
- per cache page/block, optionally with subgroups.

Fine granularity reduces error but consumes metadata and adds reductions/addressing. Static scales can be calibrated and stored in a checkpoint. Dynamic scales adapt as tokens arrive but add work and create versioning questions when old cached tokens used different scales.

Specify whether K is quantized before or after RoPE. A runtime that stores post-RoPE keys cannot consume a pre-RoPE scheme without moving rotation into attention. Cache page layout must colocate or index scales efficiently.

## Kernel implications

The useful path writes quantized K/V during append and reads/dequantizes inside attention, ideally feeding dot products without materializing full BF16 K/V. Separate kernels for prompt prefill, mixed append, and one-token decode may support different dtypes. Verify all phases.

FP8 halves element bytes relative to BF16, but scale metadata and attention compute mode vary by backend. Lower-bit integer/FP4 caches demand specialized dot kernels. Capacity gains increase concurrency only if scheduler limits and cache pools permit it.

## Implementation ladder

### 1. Reference

Implement `inference-lab/reference/quantized_kv.py`: append FP32/BF16 K/V, quantize with selectable axes, dequantize, and run a causal attention oracle. Compare token-by-token outputs and logits against an unquantized cache.

### 2. Optimized

Add a paged layout with scale sidecars and fuse dequantization into a tiled decode-attention prototype. Measure append cost, cache bytes, cache-read bandwidth, and attention latency across \(S\).

### 3. Integrated

Enable an engine’s supported FP8 KV cache with pinned checkpoint scales or an explicit scale-calculation procedure. Capture cache capacity and backend selection. Test prefill, append, decode, prefix reuse, beam/copy behavior, and mixed attention layers.

## Three 3-hour sessions

1. **Memory and numerics:** derive cache sizes for MHA/GQA/hybrid models; compare per-tensor, per-token, and K/V-asymmetric quantizers.
2. **Cache implementation:** add quantized append/read to a paged-cache simulator and verify cached-versus-full logits for every step.
3. **Runtime profile:** trace K/V scale loading, cache allocation, attention backend, and fused dequantization; sweep context and concurrency.

The weekend study includes short PPL/logit KL, long-context retrieval or needle tests at multiple depths, and real TTFT/TPOT/throughput under a fixed workload.

## Experiments and deliverables

Commit:

- `reference/quantized_kv.py`;
- `tests/test_quantized_kv_attention.py`;
- `engine/cache/quantized_pages.py`;
- `benchmarks/results/kv-precision-context.json`;
- `benchmarks/reports/kv-quality-capacity.md`;
- `notes/source_walks/kv-scale-to-attention.md`.

Record cache dtype, scale axis/dtype/source, RoPE location, page size/layout, retained window, attention backend, context distribution, concurrency, and preemption count.

## Common failure modes

- Applying the uniform-layer memory formula to GQA or hybrid attention.
- Using scale 1.0 when calibrated scales are absent and assuming accuracy is validated.
- Quantizing K on the wrong side of RoPE.
- Testing only a short perplexity set.
- Measuring memory capacity without scheduler concurrency.
- Dequantizing the whole cache into a temporary each token.
- Verifying decode but not prefill/append or prefix reuse.
- Assuming weight format determines KV-cache format.

## Mastery gate

For a supplied architecture and workload, compute exact cache payload and metadata, choose K/V scale axes, implement cached-attention equivalence tests, and show a context/concurrency sweep proving both quality and system impact.

## Canonical code paths and OSS bridge

Study [KIVI](https://arxiv.org/abs/2402.02750) with its [official code](https://github.com/jy-yuan/KIVI) and [KVQuant](https://arxiv.org/abs/2401.18079). In vLLM, inspect the [quantized KV-cache guide](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/), `CacheConfig`, attention layer scale loading, and backend kernels. Compare TensorRT-LLM’s [KV quantization options](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html) and FlashInfer paged attention. OSS opportunities include scale-metadata validation, long-context regression tests, or backend support tables.

## Primary and authoritative sources

- [KIVI](https://arxiv.org/abs/2402.02750)
- [Official KIVI repository](https://github.com/jy-yuan/KIVI)
- [KVQuant](https://arxiv.org/abs/2401.18079)
- [vLLM quantized KV cache](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/)
- [TensorRT-LLM quantization](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)

## Next

Module 28 closes the phase by treating checkpoint metadata, repacking, capability checks, kernels, compilation, and fallbacks as one end-to-end quantized operator.
