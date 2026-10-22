# Module 11 — Positional Encoding, RoPE, and Scaling

> **Phase:** 2 — Architecture Core  
> **Time:** Week 6; two 3-hour sessions plus a 2–4 hour independent lab and mastery gate  
> **Prerequisites:** Modules 03–10; complex numbers or 2-D rotations, attention shapes, and cached-versus-full decoding concepts

## Why this matters

Self-attention alone is permutation-equivariant: rearranging tokens rearranges outputs but supplies no notion of order. Rotary position embeddings (RoPE) inject position directly into queries and keys and are used by many current decoder models. A RoPE bug can survive shape tests while destroying generation. Context-extension settings add another danger: changing a base, scale factor, rotary fraction, or position convention may let a model allocate a longer cache but does not prove it can use that context.

## Outcomes

You will be able to:

- implement RoPE and prove its norm and relative-position properties;
- derive frequency, cache, and broadcast shapes;
- handle absolute position IDs correctly during chunked or token-by-token decode;
- distinguish extrapolation, position interpolation, NTK-aware scaling, and YaRN; and
- validate a checkpoint's exact RoPE configuration before measuring long-context quality.

## Mental model

Imagine every adjacent pair of query/key channels as clock hands rotating at a different speed. Position advances all clocks, but high-frequency clocks turn rapidly while low-frequency clocks track long spans. Comparing a rotated query with a rotated key cancels their shared absolute rotation, leaving a signal determined by relative distance. RoPE is therefore not an embedding vector added to the residual stream; it is a position-dependent change of coordinates applied to Q and K.

## Theory, math, and shapes

For rotary dimension `R` (even), pair channels into `R/2` complex numbers. At position $m$, frequency index $j$ uses

$$
\theta_j=b^{-2j/R},\qquad
z'_{m,j}=z_{m,j}e^{im\theta_j}.
$$

Equivalently, apply a 2-D rotation with cosine and sine. Because rotations are orthogonal, $\|R_m x\|=\|x\|$. More importantly,

$$
(R_m q)^\top(R_n k)=q^\top R_{n-m}k,
$$

so the attention dot product exposes relative displacement. With Q/K `[B,H,T,D]`, a frequency table is naturally `[T,R/2]`; implementations may duplicate it to `[T,R]` and broadcast cosine/sine as `[1,1,T,R]`. If `R<D`, concatenate untouched channels.

Two common pairing layouts exist: adjacent pairs `(0,1),(2,3),...` and half-split pairs `(0,R/2),(1,R/2+1),...`. Both implement rotations, but they are not checkpoint-compatible. Match the model code. Build angles in float32 because large position IDs and low-precision trigonometry amplify error.

For cached decoding, new token positions are absolute: if `p` tokens are already cached, a chunk of length `L` uses `[p,...,p+L-1]`. Resetting every chunk to zero breaks equivalence with full forward.

If a model trained through length $L$ is asked to serve $L'=sL$, plain extrapolation evaluates unseen phases. Position interpolation instead uses $m/s$, compressing new positions into the trained range but reducing local resolution. NTK-aware methods modify the frequency base rather than uniformly scaling every frequency. YaRN blends frequency-dependent interpolation with attention-magnitude correction. These methods are not interchangeable configuration aliases, and most benefit from—or require—the matching fine-tuned checkpoint.

## Implementation ladder

1. In `inference-lab/reference/rope.py`, implement angle generation, both pairing conventions, partial rotary dimensions, and `apply_rope(q,k,position_ids)`.
2. Add a cache for cosine/sine tables that grows monotonically by device and dtype. The cache is an optimization; direct computation remains the oracle.
3. Integrate RoPE between Q/K projection and attention in `reference/attention.py`. Never rotate V.
4. Add `RopeConfig` fields for `theta`, rotary dimension/fraction, pairing layout, original context length, scaling type, and scaling parameters. Reject incomplete combinations.
5. Implement linear position interpolation as the first scaling experiment. Treat YaRN as a source-reading exercise unless you can reproduce its complete frequency and magnitude rules.

## Two 3-hour sessions

**Session A — rotations and invariants**

- 0:00–0:40: rotate hand-worked 2-D vectors and derive the relative-dot-product identity.
- 0:40–1:20: map the derivation onto `[B,H,T,D]` tensors and partial RoPE.
- 1:20–2:15: implement direct and table-cached RoPE.
- 2:15–3:00: test norms, layouts, offsets, and full-versus-chunked equivalence.

**Session B — context scaling**

- 0:00–0:45: visualize wavelength by channel and trained versus unseen positions.
- 0:45–1:30: implement linear interpolation behind explicit configuration.
- 1:30–2:15: read PI and YaRN methods; annotate exactly which term each changes.
- 2:15–3:00: run short-context regression and long-position synthetic tests.

## Experiments and deliverables

Commit:

- `tests/test_rope.py` covering norm preservation, relative-shift invariance, both layouts, partial rotation, and cache offsets;
- `benchmarks/bench_rope.py` comparing direct trigonometry, cached tables, and fused-with-attention preparation;
- a plot in `notes/module11.md` of wavelength versus channel for two `theta` values;
- a full-forward versus irregular chunks test, for example chunk sizes `[5,1,7,3]`; and
- a scaling report comparing unscaled extrapolation and linear interpolation inside the original window and beyond it.

Use deterministic synthetic vectors before a language model. A passkey result alone is not sufficient evidence of a usable extended context.

## Common failure modes

- Rotating V, rotating Q but not K, or applying RoPE after scores are formed.
- Pairing channels differently from the checkpoint.
- Using `d_model` rather than head or configured rotary dimension to build frequencies.
- Resetting position IDs for each decode step or left-padded batch.
- Computing large-position sine/cosine in fp16.
- Reusing a table created for another device, dtype, base, or scale.
- Changing only a reported `max_position_embeddings` value: it changes validation or allocation, not learned behavior.
- Applying a generic scaling recipe to a model with custom or partial RoPE.
- Declaring success from perplexity at one length while short-context quality regresses.

## Mastery gate

Prove the relative-dot-product identity for one channel pair, then demonstrate it numerically under a shared position shift. Your implementation must match full-sequence attention when evaluated in at least four irregular chunks and must fail a deliberately reset-position test. Given a checkpoint configuration, you pass when you can state the exact rotary dimension, layout, base, position offset, and scaling rule before loading weights.

## Primary sources

- Su et al., [“RoFormer: Enhanced Transformer with Rotary Position Embedding”](https://arxiv.org/abs/2104.09864).
- Chen et al., [“Extending Context Window of Large Language Models via Position Interpolation”](https://arxiv.org/abs/2306.15595).
- Peng et al., [“YaRN: Efficient Context Window Extension of Large Language Models”](https://arxiv.org/abs/2309.00071) and the [official repository](https://github.com/jquesnelle/yarn).
- Hugging Face Transformers, [current RoPE initialization utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py).

## Code paths to read

Your paths are `reference/rope.py`, `reference/attention.py`, `tests/test_rope.py`, `benchmarks/bench_rope.py`, and `notes/module11.md`.

## OSS bridge

Map your operations to Transformers’ [`LlamaRotaryEmbedding` and `apply_rotary_pos_emb`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py). In the RoPE utilities, compare `default`, `linear`, dynamic/NTK, and YaRN initialization functions field by field. Note which values come from model config rather than runtime flags.

## Next

Module 12 turns attention into an incremental decoder with MQA/GQA and a correctly managed KV cache.
