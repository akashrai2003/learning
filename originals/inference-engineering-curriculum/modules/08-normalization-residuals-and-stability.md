# Module 08 — Normalization, Residuals, and Stability

> **Phase:** 2 — Architecture Core  
> **Time:** Week 4; two 3-hour sessions plus a 2–4 hour fusion study  
> **Prerequisites:** Modules 04 and 06, reduction operations, and mixed-precision basics

## Why this matters

Normalization is a small fraction of model FLOPs but lies on every residual boundary. A wrong epsilon, scale convention, axis, accumulation dtype, or residual order compounds through all layers. Optimized engines frequently fuse residual addition and RMSNorm, making a precise reference contract essential.

## Outcomes

You can:

- derive LayerNorm and RMSNorm and state their invariances;
- distinguish pre-norm, post-norm, and model-specific residual layouts;
- implement stable FP32-statistic references for low-precision inputs;
- define and test fused add-plus-norm semantics without aliasing bugs;
- diagnose drift using layerwise residual statistics.

## Mental model: preserve the highway, condition the branch

The residual stream is the model’s information highway. In a pre-norm block, the branch reads a normalized view and its update is added to the unnormalized stream:

\[
y=x+F(N(x)).
\]

In the original post-norm Transformer:

\[
y=N(x+F(x)).
\]

They are different architectures, not interchangeable kernel schedules. The checkpoint was trained with one exact graph.

Normalization reduces sensitivity to scale and stabilizes the values presented to learned projections. It does not guarantee that an incorrect residual stream remains harmless.

## Theory, math, and tensor shapes

For each token vector \(x:[D]\), LayerNorm computes:

\[
\mu=\frac{1}{D}\sum_i x_i,\quad
\sigma^2=\frac{1}{D}\sum_i(x_i-\mu)^2,
\]
\[
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
\]

RMSNorm omits centering:

\[
r=\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon},\qquad
y_i=\gamma_i\frac{x_i}{r}.
\]

For `x:[B,S,D]`, statistics are independently reduced over the last axis and output shape remains `[B,S,D]`. LayerNorm is invariant to a constant shift before affine terms; RMSNorm is not. Both are approximately invariant to positive scaling when epsilon is negligible.

For FP16/BF16 input, a trustworthy reference casts values to FP32 for squared reduction and reciprocal square root, then casts the normalized result as required. Production kernels may use specialized reductions but must match the declared tolerance.

A common fused contract takes residual \(r\) and update \(u\):

\[
r_{\text{new}}=r+u,\qquad n=\mathrm{RMSNorm}(r_{\text{new}}),
\]

and returns both `n` and `r_new`. Whether inputs may alias, which tensor is updated in-place, and what dtype stores `r_new` are part of the API. Some architectures use an offset scale such as \(1+\gamma\); never infer that convention from a parameter initialized near zero.

## Implementation ladder

1. Add `reference/norm.py` with explicit LayerNorm, RMSNorm, and configurable scale-offset semantics.
2. Validate last dimension, parameter shapes, finite epsilon, and supported accumulator dtype.
3. Add `reference/residual.py` with pure pre-norm and post-norm compositions.
4. Add a non-mutating `add_rms_norm_ref(update, residual)` returning normalized output and new residual.
5. Implement an optimized PyTorch version under `kernels/` only after the reference tests pass.
6. Add `tests/test_norm.py` comparing PyTorch primitives, your equations, and the optimized path across dtypes, awkward \(D\), non-contiguous input, and extreme values.
7. Instrument the tiny decoder to record per-layer residual RMS, mean, max magnitude, and fraction non-finite.

## Two 3-hour sessions

**Session A — equations and topology**

- **0:00–0:35 — Derivation:** calculate LayerNorm and RMSNorm on a four-element vector.
- **0:35–1:05 — Architecture:** trace pre-, post-, and fused residual layouts.
- **1:05–1:15 — Break.**
- **1:15–1:55 — Reference lab:** implement FP32-statistic norms and invariance tests.
- **1:55–2:30 — Fusion lab:** specify aliasing and compare fused versus unfused results.
- **2:30–3:00 — Stability lab:** inject epsilon, axis, and scale-offset bugs and localize first-layer drift.

**Session B — precision and fusion evidence**

- **0:00–0:45:** sweep dtype, epsilon, hidden width, scale, and shift.
- **0:45–1:30:** implement and test the optimized add-plus-RMSNorm path.
- **1:30–2:15:** inspect memory traffic and benchmark fused versus unfused execution.
- **2:15–3:00:** run layerwise drift experiments and finalize the stability report.

## Experiments and deliverables

Submit reference and optimized functions, tests, and `notes/norm-stability.md`. Sweep hidden width, epsilon, input scale, shift, dtype, and accumulator policy. Plot max and RMS error.

Run the tiny decoder for at least 12 layers with fixed random weights. Compare correct execution with three injected faults: low-precision reduction, epsilon outside the square root, and swapped residual order. Record the first layer exceeding a declared error threshold rather than only final logits.

Benchmark separate add plus RMSNorm against a fused implementation. Count HBM reads/writes and explain speedup through data movement, not fewer normalization equations.

## Failure modes

- Reducing across batch or sequence instead of hidden width.
- Computing squares and sums in FP16 and overflowing.
- Placing epsilon outside `sqrt` or using checkpoint-incompatible epsilon.
- Treating RMSNorm as mean-centered LayerNorm.
- Applying affine bias where the architecture has none.
- Missing a model’s unit-offset scale convention.
- Returning a normalized tensor where the next residual addition expects the unnormalized stream.
- In-place mutation that corrupts a cache or graph-captured input.

## Mastery gate

Derive both norms, predict their response to shift and scaling, and specify a fused residual API including alias and dtype rules. Reference, framework, and optimized outputs must pass edge-case parity, and layerwise instrumentation must localize every injected fault to its first affected block.

## Primary sources

- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Root Mean Square Layer Normalization](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
- [Canonical Llama RMSNorm implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

## Code paths to read

- vLLM: `vllm/model_executor/layers/layernorm.py`
- Transformers: `src/transformers/models/llama/modeling_llama.py`
- PyTorch: `torch/nn/modules/normalization.py`
- Triton: `python/tutorials/05-layer-norm.py`

## OSS bridge

Compare your fused contract with vLLM’s custom RMSNorm operations and identify what each return value represents. Inspect a real checkpoint’s epsilon, affine parameters, and scale convention. Add an adapter test; do not hard-code one model family’s convention into the general reference.

## Next

Module 09 studies the other per-token branch—gated feed-forward networks—and turns its three projections and activation into an optimization target.
