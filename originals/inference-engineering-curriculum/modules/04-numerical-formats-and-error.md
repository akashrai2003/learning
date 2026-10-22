# Module 04 — Numerical Formats and Error

> **Phase:** 1 — Foundations  
> **Time:** Week 2; two 3-hour sessions plus a 2–4 hour quantization report  
> **Prerequisites:** Module 03, binary representation, logarithms, and basic probability

## Why this matters

Lower precision can reduce weight traffic and unlock Tensor Cores, but “four bit” is not a complete numerical specification. Format, scaling granularity, zero/NaN behavior, accumulation dtype, rounding, layout, and hardware support jointly determine accuracy and speed. An inference engineer must predict failure before a benchmark merely returns fluent nonsense.

## Outcomes

You can:

- compare FP32, TF32, FP16, BF16, FP8 E4M3/E5M2, integers, MXFP4, and NVFP4;
- separate storage, operand, accumulator, and output precision;
- derive quantize/dequantize operations and block-scale shapes;
- measure saturation, underflow, bias, MSE, relative error, and task-level drift;
- design reference tests with defensible tolerances.

## Mental model: payload plus metadata plus compute rule

A low-precision tensor is often not one array. It is packed payload \(q\), scale metadata \(s\), perhaps zero-points \(z\), and a layout contract interpreted by a particular kernel:

\[
\hat{x}=s(q-z).
\]

Per-tensor scales are cheap but vulnerable to outliers. Per-channel or block scales track local dynamic range at the cost of metadata and layout work. Compression only improves wall time if the target hardware consumes the representation without expensive dequantization or rearrangement.

## Theory, math, and shapes

A normalized binary floating-point value is approximately

\[
(-1)^{b_s}\,2^{e-\mathrm{bias}}\,(1.f).
\]

Exponent bits buy range; fraction bits buy local precision. FP16 has more fraction precision but far less range than BF16. FP8 E4M3 favors precision, while E5M2 favors range. Products may use narrow operands and FP32 accumulation; saying “FP16 matmul” without the accumulator is incomplete.

For round-to-nearest with unit roundoff \(u\), one operation is modeled by

\[
\mathrm{fl}(x\circ y)=(x\circ y)(1+\delta),\quad |\delta|\le u.
\]

A naive length-\(n\) accumulation has a bound involving
\(\gamma_n=nu/(1-nu)\) when \(nu<1\). It is a warning model, not a promise: cancellation, reduction order, fused multiply-add, subnormals, and overflow matter.

For symmetric block quantization of \(X:[M,K]\) with block length \(G\),
scales may be `[M, ceil(K/G)]`, while packed 4-bit payload occupies roughly `M*K/2` bytes. OCP MX formats commonly share a power-of-two scale over 32 values. NVIDIA NVFP4 uses E2M1 payload, a local E4M3 scale for 16 values, and a global FP32 scale:

\[
\hat{x}=q_{\mathrm{E2M1}}\,s_{\mathrm{E4M3,block}}\,s_{\mathrm{FP32,global}}.
\]

Metadata, padding, and swizzled scale layouts must be included in actual byte counts.

## Implementation ladder

1. Build `reference/formats.py` with `torch.finfo` reports and explicit emulators for symmetric INT8, E4M3-like clipping/rounding, and block-scaled E2M1.
2. Keep quantization and dequantization separate. Return payload, scale, original shape, padding, and format name.
3. Add `reference/error_metrics.py`: absolute/relative error quantiles, MSE, cosine similarity, saturation rate, nonzero-to-zero underflow rate, and bias.
4. Test distributions with Gaussian values, heavy tails, controlled outliers, all zeros, tiny magnitudes, and non-finite values.
5. Apply quantization to `[M,K] @ [K,N]`; accumulate dequantized products in FP32 and compare with a CPU FP64 reference.
6. Add `tests/test_formats.py` for representable values, ties, scale-zero blocks, padding, NaNs, and deterministic rounding.

The emulator teaches semantics; it is not expected to be fast or bit-identical to every hardware encoding.

## Two 3-hour sessions

**Session A — formats and error**

- **0:00–0:40 — Format anatomy:** enumerate representative values, gaps, maxima, minima, infinities, and NaNs.
- **0:40–1:10 — Error model:** derive unit roundoff and demonstrate non-associative reduction.
- **1:10–1:20 — Break.**
- **1:20–2:00 — Quantizer lab:** implement per-tensor and block scaling with explicit metadata.
- **2:00–2:35 — GEMM experiment:** sweep block size and accumulator dtype.
- **2:35–3:00 — Hardware bridge:** inspect a block-scaled Triton matmul and distinguish logical scale shape from packed layout.

**Session B — falsify a precision policy**

- **0:00–0:45:** build edge-case and distribution generators.
- **0:45–1:30:** sweep scale granularity, rounding, and accumulator dtype.
- **1:30–2:15:** compare operator metrics with logit top-1 and KL drift.
- **2:15–3:00:** account for metadata/layout bytes and write the precision recommendation.

## Experiments and deliverables

Submit the emulator, edge-case tests, and `notes/precision-report.md`. Sweep format, block size, distribution, and reduction length. Plot error quantiles and saturation—not only mean error. Include effective bytes per parameter with scales and padding.

Choose a small linear layer from the cumulative stack and compare final-logit top-1 agreement and KL divergence after quantizing its weights. State clearly that layer error is not a substitute for end-task evaluation.

## Failure modes

- Equating bit width with format or kernel support.
- Ignoring accumulator precision and reduction order.
- Choosing a scale from `amax=0` and dividing by zero.
- Reporting only MSE while rare saturated outliers change argmax.
- Dequantizing to BF16 in HBM and claiming 4-bit bandwidth.
- Forgetting scale metadata, padding, or packing overhead.
- Using relative error near zero without an absolute-error companion.
- Assuming mathematically equivalent GPU reductions are bitwise equal.

## Mastery gate

Given a tensor distribution and target format, select scale granularity, predict dominant error, and design an experiment that could falsify the choice. Your implementation must round-trip edge cases, expose saturation/underflow, and explain any tolerance from operation count and dtype rather than copying a magic constant.

## Primary sources

- [CUDA floating-point computation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)
- [NVIDIA Transformer Engine NVFP4 format](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)

## Code paths to read

- Triton: `python/tutorials/10-block-scaled-matmul.py`
- Transformer Engine: `transformer_engine/common/recipe/__init__.py`
- FlashInfer: `flashinfer/gemm.py` and FP4 JIT generators
- PyTorch: `torch/_inductor/kernel/mm.py`

## OSS bridge

Compare your logical scale arrays with Triton’s block-scaled tutorial and Transformer Engine’s required rowwise/columnwise layouts. Then inspect how an inference engine’s quantization config selects a linear kernel. Configuration describes representation; the log-selected kernel proves execution.

## Next

Module 05 combines FLOPs and bytes into a roofline model so precision choices become performance hypotheses rather than slogans.
