# Module 04 — Tensor Math and Numerical Contracts

> **Phase:** I — Engineering foundations  
> **Time:** Week 2; three focused 3-hour sessions  
> **Prerequisites:** Modules 01–03, matrix multiplication, probability, and logarithms

## Existing inference curriculum bridge

Pair this with [Module 03 — `tensor-math-shapes-and-einsum.md`](../../inference-engineering-curriculum/modules/03-tensor-math-shapes-and-einsum.md), [Module 04 — `numerical-formats-and-error.md`](../../inference-engineering-curriculum/modules/04-numerical-formats-and-error.md), [Module 18 — `quantization-fundamentals.md`](../../inference-engineering-curriculum/modules/18-quantization-fundamentals.md), and [Module 19 — `calibration-error-metrics-and-evaluation.md`](../../inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md). Here you build PyTorch reference implementations that those inference modules will later quantize and lower.

## Why this matters and outcomes

A tensor program can have legal shapes and still be wrong: reduction axes can be misplaced, logits can be normalized over batch rather than vocabulary, a causal mask can be inverted, or half-precision accumulation can turn a correct formula into NaNs. The only reliable optimization baseline is a numerical contract: exact semantics where possible and a justified tolerance where not.

You will be able to derive shape and dtype contracts for common model operations, use PyTorch’s broadcasting and reduction semantics deliberately, implement stable softmax/cross entropy/norms, choose an error metric that matches a claim, and write randomized differential tests against a simple oracle.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P04 — Tensor math and numerical contracts**.

1. [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — learn the API or system contract.
2. [Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — study the underlying design, paper, or specification.
3. [PyTorch tensor documentation](https://docs.pytorch.org/docs/stable/tensors.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [Python data model](https://docs.python.org/3/reference/datamodel.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Python profiling](https://docs.python.org/3/library/profile.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: tensors are indexed functions with representation

A tensor `x` is not merely a byte buffer. It represents a function over named index sets plus dtype, device, layout, and aliasing rules. Start every expression by writing axes, not dimensions. For a decoder:

| Symbol | Meaning | Typical tensor |
|---|---|---|
| `B` | batch/active sequences | `input_ids: int64[B,T]` |
| `T` | token positions | `hidden: float[B,T,D]` |
| `D` | model width | weights use `[out,in]` in `nn.Linear` |
| `H, Dh` | heads and head width | `q: [B,H,T,Dh]` |
| `V` | vocabulary | `logits: [B,T,V]` |

Semantic labels matter: `[B,T]` masks, targets, and IDs are not interchangeable. PyTorch can broadcast an error without warning, so build assertions around both names and values.

## Theory, math, and numerical contracts

For linear `y=xW^T+b`, PyTorch `nn.Linear(in_features, out_features)` stores `weight:[out,in]`; with `x:[...,in]`, output is `[...,out]`. For a batched matrix product, `torch.matmul(A:[B,M,K], B:[B,K,N]) -> [B,M,N]`. `einsum('btd,df->btf', x, w)` makes contracted and retained axes explicit; it is a specification tool before it is a performance choice.

Softmax is

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

Computing `exp(z)` directly overflows. Subtracting the row maximum preserves the result:

\[
\operatorname{softmax}(z)_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\quad m=\max_jz_j.
\]

For class target \(y\), negative log likelihood is \(-z_y+\log\sum_j e^{z_j}\), which is why `torch.nn.functional.cross_entropy` consumes unnormalized logits rather than manually softmaxed probabilities. A probability zero may be valid; `log(0)` is not. Use the framework’s stable fused primitive unless a reference implementation is being taught/tested.

Floating-point arithmetic rounds. A useful model is \(\mathrm{fl}(a\circ b)=(a\circ b)(1+\delta)\), \(|\delta|\le u\), where \(u\) depends on format and operation. Reduction order changes results, especially in fp16/bf16. An operand format, accumulation dtype, output dtype, and tolerances are therefore one contract. `torch.allclose` with arbitrary defaults is not one: use absolute tolerance near zero and relative tolerance elsewhere, check NaN/Inf explicitly, and make tolerance proportional to expected reduction depth.

For comparison vectors \(a,b\), choose among max absolute error \(\max_i|a_i-b_i|\), relative error when the reference magnitude is material, cosine similarity for direction, and logit/KL or top-k agreement for model behavior. Do not use cosine similarity to excuse a one-hot classification error, or text equality to prove transformer parity.

## Implementation ladder in the proposed `torch-lab/`

1. **Reference:** implement `ops/reference_math.py`: stable `softmax`, RMSNorm, cross entropy, and `scaled_dot_product_attention_reference`. Each function states axes, allowed dtypes, accumulation dtype, and output contract. Use small explicit loops only as an oracle.
2. **Optimized:** replace reference paths selectively with `torch.nn.functional` primitives and `torch.matmul`/`einsum`; retain the oracle tests. Benchmark shapes `[B,T,D]` representative of tiny training, prefill, and one-token decode rather than a single square matrix.
3. **Integrated:** make one decoder block consume the contract functions behind an implementation selector (`reference`/`torch`). Log maximum error, NaN count, and chosen dtype per test case; later modules will add compiled and quantized implementations without changing test inputs.

## Three focused sessions

1. **Session A — derive before coding (3h):** hand-derive linear, layer/RMS norm, log-softmax, and cross entropy shapes. Implement each with a small CPU oracle and tests that intentionally reduce over the wrong axis.
2. **Session B — precision (3h):** compare fp32, bf16, and fp16 where supported. Sweep sequence length/reduction size; collect max/mean errors, NaN/Inf counts, and the effect of fp32 accumulation. Explain why two valid reductions differ bitwise.
3. **Session C — contracts (3h):** create property tests over non-square, singleton, and ragged-mask-like padded shapes. Include adversarial logits around ±100 and all-masked rows; decide and document the desired behavior instead of accepting accidental NaNs.

## Experiments and deliverables

Deliver `tests/numerics/test_reference_math.py`, a tolerance table keyed by dtype/operation/shape, and a memo that distinguishes mathematical error from implementation bug. Establish one golden fp32 fixture saved as tensors plus metadata, not a screenshot. Compare PyTorch primitives to your reference on CPU first, then CUDA; differences across backends should be measured, not assumed bugs.

## Common failure modes

- Calling `softmax` before `cross_entropy`, effectively changing the loss contract.
- Permuting `T` and `H` or reducing `dim=0` because a small square test hid the error.
- Creating a float mask/target through implicit type promotion instead of retaining `bool`/`int64` semantics.
- Asking for bitwise equality across parallel reductions, devices, or low-precision Tensor Core paths.
- Applying a tolerance large enough to conceal a missing scale, wrong mask polarity, or wrong output projection.

## Mastery gate

Implement an attention-like operator from the written equations, then pass randomized tests against PyTorch across `B∈{1,3}`, unequal `Tq,Tk`, fp32 and a supported reduced precision. Diagnose three seeded failures: axis swap, unstable softmax, and incorrect accumulation. Defend each tolerance and state when equality should be exact.

## Primary sources and canonical code paths

- [PyTorch tensor semantics](https://docs.pytorch.org/docs/stable/tensors.html), [`torch.matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html), [`torch.einsum`](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html), and [numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
- [PyTorch `cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html), [scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), and [`torch.testing`](https://docs.pytorch.org/docs/stable/testing.html).
- Inspect [`aten/src/ATen/native/SoftMax.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SoftMax.cpp), [`aten/src/ATen/native/LossNLL.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossNLL.cpp), and their tests under [`test/`](https://github.com/pytorch/pytorch/tree/main/test).

## OSS bridge and next

Numerical regression tests are a major form of PyTorch contribution: they document edge cases as well as prevent regressions. Keep the reference paths small, readable, and independent of the optimized path. Next, learn how tests and measurements make those contracts trustworthy in [Module 05](05-testing-debugging-and-benchmark-discipline.md).
