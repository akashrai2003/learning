# Module 29 — Attention IO Complexity and Online Softmax

**Phase:** 5 — Attention and decoding systems  
**Pacing:** Week 15, paired with Module 30; three 3-hour sessions plus a weekend tiled-attention proof  
**Prerequisites:** Modules 05, 10, 12, and 27; roofline analysis, causal attention, GQA/KV cache, and numerical stability

## Why this matters

Dense attention’s arithmetic is quadratic in sequence length, but conventional implementations also materialize and reread a quadratic score/probability matrix. Modern exact-attention kernels are fast because they reorganize memory traffic, not because they approximate the equation. Online softmax is the algebraic tool that lets key/value tiles be consumed without storing all scores.

## Outcomes

You can:

- derive attention FLOPs, intermediate bytes, and arithmetic intensity;
- implement numerically stable online softmax;
- combine partial softmax states and value numerators exactly;
- build a tiled exact-attention reference with causal/GQA masks;
- explain what FlashAttention changes—and what remains quadratic.

## Mental model: preserve three running statistics

For one head,

\[
S=QK^\top/\sqrt D,\qquad P=\operatorname{softmax}(S),\qquad O=PV.
\]

With \(Q[B,H_q,S_q,D]\), \(K,V[B,H_{kv},S_k,D]\), GQA maps each query head to one KV head. Score and probability tensors are logically
\([B,H_q,S_q,S_k]\).

The two matrix multiplications cost approximately

\[
4BH_qS_qS_kD
\]

FLOPs in total. Standard attention writes and reads \(O(BH_qS_qS_k)\) scores/probabilities in HBM. Tiled exact attention keeps score tiles on chip, so external storage becomes linear in Q/K/V/O size, although mathematical work remains \(O(S_qS_kD)\).

Stable row softmax normally computes

\[
m=\max_i x_i,\quad \ell=\sum_i e^{x_i-m},\quad
p_i=e^{x_i-m}/\ell.
\]

For tiled attention, also maintain the unnormalized value numerator

\[
o=\sum_i e^{x_i-m}v_i.
\]

Suppose accumulated state is \((m,\ell,o)\) and a new score/value tile has local state \((m_b,\ell_b,o_b)\). Combine them with

\[
m'=\max(m,m_b),
\]

\[
\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b,
\qquad
o'=e^{m-m'}o+e^{m_b-m'}o_b.
\]

After all tiles, \(O=o/\ell\). This recurrence is associative up to floating-point rounding, enabling streaming and parallel reductions.

## Masks and edge cases

Apply scale, bias, and causal/local mask to each score tile before its max. Masked entries are \(-\infty\), not a large arbitrary negative value that may behave differently by dtype. An all-masked row yields \(m=-\infty\) and requires a defined zero/error policy to avoid \(-\infty-(-\infty)\). Align causal coordinates correctly when \(S_q\ne S_k\), as in decode or chunked prefill.

Accumulate \(m,\ell\), dot products, and output in explicitly chosen precision. Different tile orders change rounding; compare error with a high-precision reference rather than requiring bit identity.

## IO analysis

Build two cost models:

1. **Materialized:** read Q/K, write S, read/write softmax probabilities, read P/V, write O.
2. **Tiled:** read Q tiles, stream K/V tiles, retain running state on chip, write O.

K/V may be reread across Q tiles, so tile sizes and on-chip capacity matter. The original FlashAttention paper derives an IO bound as a function of SRAM size; “never materializes \(S\)” is necessary but not a complete performance model.

Prefill has large \(S_q\) and exposes parallel tiles. Decode has \(S_q=1\), so a different attention kernel and paged KV layout are often appropriate.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/online_softmax.py` with scalar, block, and merge implementations. Add `reference/tiled_attention.py` that tiles over Q and KV while retaining FP64/FP32 running states.

### 2. Optimized

Vectorize tiles or implement a Triton prototype. Fuse scale, mask, online softmax, and \(PV\). Parameterize block sizes and record estimated versus profiler-observed HBM bytes.

### 3. Integrated

Compare the prototype with framework SDPA and a production attention backend across causal/noncausal, GQA, variable length, local window, and decode/prefill shapes. Capture the backend actually selected.

## Three 3-hour sessions

1. **Derive:** calculate FLOPs/bytes for materialized attention and prove the online-state merge equations.
2. **Implement:** write scalar/block softmax and tiled exact attention; test random partitions and extreme logits.
3. **Profile:** sweep sequence/head dimensions and tile sizes, inspect memory traffic, and trace framework attention dispatch.

The weekend proof includes equations, property tests, profiler screenshots/data, and one counterexample where a theoretically lower-IO tile performs worse because of occupancy or overhead.

## Experiments and deliverables

Commit:

- `reference/online_softmax.py`;
- `reference/tiled_attention.py`;
- `tests/test_online_softmax.py`;
- `benchmarks/results/attention-io-tiles.json`;
- `notes/source_walks/attention-dispatch.md`.

Test arbitrary partitions, large positive/negative logits, all-masked rows, causal offsets, non-power-of-two lengths, GQA head mapping, and FP16/BF16 inputs against an FP64 oracle.

## Common failure modes

- Saying FlashAttention reduces attention’s arithmetic complexity.
- Combining block sums without rescaling from old maxima.
- Normalizing each tile independently and then averaging outputs.
- Using mask coordinates valid only for \(S_q=S_k\).
- Treating \(-10^4\) as universally equivalent to \(-\infty\).
- Measuring allocated tensor memory instead of actual HBM traffic.
- Generalizing a prefill result to one-token decode.

## Mastery gate

Derive the three-state recurrence from stable softmax, implement partition-invariant tiled attention, and explain its FLOPs and HBM traffic. Pass causal/GQA/extreme-logit tests and reconcile profiler traffic with the cost model.

## Canonical code paths and OSS bridge

Read PyTorch SDPA dispatch and the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention) beside vLLM/FlashInfer prefill and decode attention paths. Inspect shapes, strides, mask semantics, accumulator types, and backend capability checks. Good OSS tasks include edge-mask tests, clearer dispatch diagnostics, or a benchmark shape missing from CI.

## Primary sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Official FlashAttention implementation](https://github.com/Dao-AILab/flash-attention)

## Next

Module 30 follows the FlashAttention line from IO-aware tiling through improved work partitioning, Hopper asynchrony, FP8, and Blackwell pipeline co-design.
