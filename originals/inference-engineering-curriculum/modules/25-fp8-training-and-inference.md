# Module 25 — FP8 Training and Inference

**Phase:** 4 — Quantization  
**Pacing:** Week 13, paired with Module 26; three 3-hour sessions plus a weekend FP8 recipe report  
**Prerequisites:** Modules 03–05 and 18–24; floating-point formats, scaling, calibration, GEMM shapes, and evaluation

## Why this matters

FP8 preserves floating-point dynamic range more gracefully than INT8 while halving BF16/FP16 storage and enabling native tensor-core operations on supported hardware. It is nevertheless not “cast everything to eight bits.” E4M3/E5M2 choice, scale granularity, amax timing, accumulation, communication, and operator coverage determine stability and speed.

## Outcomes

You can:

- decode E4M3 and E5M2 range/precision trade-offs;
- implement scaled FP8 emulation with explicit rounding and saturation;
- distinguish current, delayed, static, and dynamic scaling;
- explain hybrid FP8 training and FP8 post-training inference;
- verify that an engine uses native scaled GEMM rather than conversion fallback.

## Mental model: a floating format plus an external scale

FP8 encodings contain one sign bit and either:

- **E4M3:** four exponent and three mantissa bits; more precision, less range;
- **E5M2:** five exponent and two mantissa bits; more range, less precision.

Exact special-value and maximum-value behavior depends on the standardized variant. NVIDIA Transformer Engine’s E4M3 description gives maximum magnitude 448, while E5M2 reaches 57,344. Pin the dtype (`float8_e4m3fn`, `float8_e5m2`, and so on), not only “FP8.”

For real tensor \(X\), store \(Q=\operatorname{cast}_{FP8}(X/s)\) and reconstruct

\[
\hat X=sQ.
\]

Equivalently some APIs store an inverse scale \(r=1/s\) and cast
\(\operatorname{cast}(rX)\). State the convention. With amax scaling,

\[
r=\frac{F_{\max}}{\max|X|},
\]

then values outside range saturate and tiny values may underflow. Tensorwise, rowwise, and blockwise scales have different metadata and kernel interfaces.

For \(A[M,K]\) and \(B[K,N]\), scaled GEMM conceptually computes

\[
C \approx (s_A Q_A)(s_B Q_B),
\]

with higher-precision accumulation and an output dtype/scale chosen separately. Scale tensors must broadcast in exactly the pattern supported by the hardware kernel.

## Scaling recipes

**Current scaling** computes current amax, derives a scale, then casts. It adapts immediately but usually reads the tensor once for the reduction and again for conversion.

**Delayed scaling** predicts the next scale from an amax history, allowing conversion in one data pass while updating history for future steps:

\[
r_t = F_{\max}/\operatorname{amax\_reduce}(h_{t-1}),\qquad
h_t\leftarrow h_{t-1}\cup\{\max|X_t|\}.
\]

It saves traffic but stale scales can clip after distribution shifts. Distributed training must reduce amax over every rank that shares the logical tensor, and all ranks must execute collectives consistently.

**Static PTQ scales** come from calibration. **Dynamic inference scales** use each runtime tensor or token. Evaluate conversion cost in decode.

Training commonly uses E4M3 for forward weights/activations and E5M2 for gradients, while master weights, reductions, sensitive operations, and accumulation stay higher precision. Loss scaling and FP8 tensor scaling solve different problems.

## Implementation ladder

### 1. Reference

Build `inference-lab/reference/fp8.py`: bit/range table, RTNE emulation or trusted dtype cast, saturation/underflow counters, tensorwise scaling, and scaled matmul with FP32 accumulation. Compare against an FP64 oracle.

### 2. Optimized

Add per-row/current and delayed-scaling state. Benchmark amax+cast separately from GEMM for decode and prefill shapes. On capable hardware, compare emulation with the framework’s scaled-matmul primitive and Transformer Engine.

### 3. Integrated

Run a small training step or PTQ checkpoint through an official FP8 recipe. Record which layers remain BF16, scale tensors and their shapes, amax synchronization, selected GEMM kernel, and graph fusions. Evaluate baseline and FP8 loss/PPL plus system metrics.

## Three 3-hour sessions

1. **Format lab:** enumerate representable values near zero, one, and maximum; plot relative spacing, clipping, and underflow for E4M3/E5M2.
2. **Scaling lab:** implement current and delayed scaling, inject a distribution spike, and quantify stale-scale behavior.
3. **Kernel/integration:** profile scaled GEMM and conversion, inspect Transformer Engine’s recipe and an inference runtime’s FP8 loader/dispatch.

The weekend report compares BF16, E4M3, E5M2, and a hybrid recipe over numerical error, loss/PPL, bytes, conversion cost, and GEMM throughput.

## Experiments and deliverables

Commit:

- `reference/fp8.py`;
- `tests/test_fp8_values_and_scales.py`;
- `benchmarks/results/fp8-shape-scale-sweep.json`;
- `benchmarks/reports/fp8-recipe.md`;
- `notes/source_walks/fp8-scaled-gemm.md`.

Record GPU architecture, dtype spelling, rounding/saturation policy, scale convention, accumulator/output dtype, warm-up, and whether scales were calibrated or computed online.

## Common failure modes

- Treating every E4M3 encoding as bit-identical.
- Mixing scale with inverse scale.
- Casting before calculating amax and hiding saturation.
- Using one global scale for tensors with unrelated distributions.
- Reducing amax over the wrong distributed group.
- Reporting tensor-core peak without scale/conversion overhead.
- Assuming checkpoint FP8 means all operators and KV cache are FP8.
- Comparing loss/PPL with different seeds, batches, or token counts.

## Mastery gate

Given a tensor distribution and target FP8 kernel, choose format, granularity, and timing of scales; derive the quantize/dequantize path; reproduce saturation/underflow counts; and show numerical plus profiler evidence for one complete FP8 GEMM.

## Canonical code paths and OSS bridge

Read the format paper beside Transformer Engine’s [current-scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) and [delayed-scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html) implementations. Trace recipe → quantizer → fused linear → scaled GEMM. Compare TorchAO’s FP8 workflows and your engine’s FP8 quantization config. Useful OSS work includes scale-convention documentation, edge-value tests, saturation telemetry, and small-\(M\) benchmarks.

## Primary and authoritative sources

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [Transformer Engine FP8 current scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html)

## Next

Module 26 shrinks element precision again and compensates with local shared exponents/scales: OCP MXFP4 and NVIDIA NVFP4.
