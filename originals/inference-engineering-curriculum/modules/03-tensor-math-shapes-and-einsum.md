# Module 03 — Tensor Math, Shapes, and Einstein Notation

> **Phase:** 1 — Foundations  
> **Time:** Week 2; two 3-hour sessions plus a 2–4 hour property-test lab  
> **Prerequisites:** Modules 01–02, linear algebra, Python, and array indexing

## Why this matters

Inference bugs frequently produce plausible numbers with the wrong semantics: a transpose swaps heads and sequence, a broadcast materializes a giant intermediate, or a view silently changes meaning because the tensor is non-contiguous. Shape literacy is therefore both a correctness discipline and a performance skill. Einstein notation makes contraction intent explicit before optimization obscures it.

## Outcomes

You can:

- annotate tensors using a stable dimension vocabulary;
- derive outputs and reduction axes from an einsum equation;
- distinguish logical shape, storage stride, view, broadcast, and materialization;
- rewrite loops as einsum, batched matmul, and flattened GEMM;
- create property tests that catch transpose and broadcasting errors.

## Mental model: named axes before numbers

Use names consistently:

- \(B\): batch or active sequences
- \(S\): query/prompt sequence length
- \(T\): flattened tokens, often \(B S\)
- \(D\): model width
- \(H\): query heads
- \(H_{kv}\): key/value heads
- \(D_h\): head width
- \(F\): feed-forward width
- \(V\): vocabulary

Shape is a logical index set. Stride says how moving one unit on an axis changes the storage address. A transpose usually changes metadata, not bytes; a later kernel may require contiguous storage and force a copy. Broadcasting is a logical expansion with a zero stride, but an operation can still write a fully materialized result.

## Theory, equations, and shapes

For matrix multiplication,

\[
C_{mn}=\sum_k A_{mk}B_{kn},
\qquad
\texttt{einsum("mk,kn->mn", A, B)}.
\]

An index absent from the output is reduced. An output index must appear in an input. Repeated free indices align; an ellipsis represents broadcast batch axes.

A token projection is:

\[
X:[B,S,D],\quad W:[D,F],\quad
Y_{bsf}=\sum_d X_{bsd}W_{df},
\]

or `einsum("bsd,df->bsf", X, W)`. A GEMM implementation commonly views \(X\) as `[T,D]`, computes `[T,D] @ [D,F] → [T,F]`, then restores `[B,S,F]`.

Attention scores use:

\[
Q:[B,H,S_q,D_h],\ K:[B,H,S_k,D_h],
\quad A_{bhqk}=\sum_d Q_{bhqd}K_{bhkd}.
\]

The result `[B,H,S_q,S_k]` is a dangerous quadratic intermediate. Later, an IO-aware kernel preserves the same mathematical contraction without storing all of it.

Count dense GEMM work as approximately \(2MKN\) floating-point operations. Count data movement separately; a view has nearly zero payload bytes, while `.contiguous()` copies every element.

## Implementation ladder

1. In `reference/tensor_contracts.py`, define checked helpers for `linear`, `split_heads`, `merge_heads`, attention scores, and weighted values.
2. Implement each operation first with scalar loops on tiny NumPy arrays.
3. Add an einsum version with named equations.
4. Add a PyTorch `matmul`/`bmm` version that reflects likely production lowering.
5. Compare all versions using random and hand-constructed tensors. Include \(B=1\), \(S_q=1\), non-square lengths, and \(H_{kv}\ne H\) placeholders.
6. In `benchmarks/layout_cost.py`, time a transpose alone, transpose plus elementwise operation, and transpose plus `.contiguous()`. Report strides and allocated bytes.
7. Add `tests/test_tensor_contracts.py` with metamorphic checks: identity weights, permutation followed by inverse permutation, and loop/einsum agreement.

Do not accept a function whose docstring says only “input tensor.” Every boundary must state axis order, dtype, device, and contiguity assumptions.

## Two 3-hour sessions

**Session A — algebra and contracts**

- **0:00–0:35 — Index algebra:** translate four loop nests into einsum and identify reductions.
- **0:35–1:10 — Transformer shapes:** derive linear, QKV split, score, value aggregation, and unembedding shapes.
- **1:10–1:20 — Break.**
- **1:20–2:00 — Reference lab:** implement loops and einsum with assertions.
- **2:00–2:30 — Layout lab:** inspect storage offsets, strides, views, broadcasts, and forced copies.
- **2:30–3:00 — Lowering lab:** replace einsum with GEMM/batched GEMM and benchmark awkward dimensions.

**Session B — adversarial shape testing**

- **0:00–0:45:** implement loop, einsum, and GEMM forms for all core contractions.
- **0:45–1:30:** generate prime-sized and non-contiguous test cases.
- **1:30–2:15:** measure implicit copies and broadcasted intermediates.
- **2:15–3:00:** inject axis bugs, localize them, and publish the tensor-contract table.

## Experiments and deliverables

Submit the checked tensor-contract library, tests, and a layout report. For at least six contractions, include equation, input/output shapes, FLOP estimate, logical broadcast axes, and whether the implementation allocates an intermediate.

Create one intentionally wrong attention transpose that still runs when \(S=H\). Show why varied test dimensions catch it while symmetric “toy” shapes do not.

## Failure modes

- Using `view` after transpose without establishing compatible strides.
- Relying on \(B=1\) or equal dimensions, which hides axis swaps.
- Confusing elementwise `*` with matrix `@`.
- Expanding K/V heads by copying when a grouped mapping suffices.
- Assuming broadcasting is always free or always materialized.
- Comparing only output shape rather than element semantics.
- Timing an asynchronous operation or including one-time compilation.

## Mastery gate

Given a contraction expressed as loops, produce the einsum, output shape, reduction axes, FLOPs, and a likely GEMM lowering. Given a tensor’s shape and strides, determine whether a reshape is a view. All reference and metamorphic tests must pass with unequal prime-sized dimensions.

## Primary sources

- [PyTorch `einsum`](https://docs.pytorch.org/docs/stable/generated/torch.functional.einsum.html)
- [NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html)

## Code paths to read

- PyTorch: `torch/functional.py` (`einsum`)
- nanoGPT: `model.py`, especially QKV reshape and transpose
- Transformers: `src/transformers/models/llama/modeling_llama.py`
- Triton: `python/tutorials/03-matrix-multiplication.py`

## OSS bridge

Trace one Llama decoder layer in Transformers and write a shape comment beside each operation without editing upstream. Then find the corresponding flattened-token contracts in a vLLM model implementation. Differences in layout are implementation choices; the contraction semantics must agree.

## Next

Module 04 adds dtype and error contracts to these shape contracts, preparing you to reason about mixed precision and quantization.
