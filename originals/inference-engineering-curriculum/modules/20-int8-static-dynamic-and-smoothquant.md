# Module 20 — INT8: Static, Dynamic, and SmoothQuant

**Phase:** 4 — Quantization  
**Pacing:** Week 10, paired with Module 19; three 3-hour sessions plus a weekend W8A8 study  
**Prerequisites:** Modules 18–19; linear algebra, observers, calibration, and quantized evaluation

## Why this matters

INT8 is mature hardware territory, but LLM activations contain persistent channel outliers. Static scales are cheap at runtime yet can be brittle; dynamic scales adapt but add reductions and conversions. SmoothQuant makes W8A8 viable by moving quantization difficulty from activations into weights through an exact offline transformation. Deployment success depends as much on kernel fusion and scale granularity as on the paper algorithm.

## Outcomes

You can:

- implement static and dynamic INT8 linear paths;
- derive zero-point correction and accumulation requirements;
- implement and verify SmoothQuant’s equivalent transformation;
- choose calibration and smoothing parameters using measured quality;
- determine whether a runtime executes native INT8 or quantize/dequantize fallbacks.

## Mental model: when are qparams chosen?

For \(X\in\mathbb{R}^{M\times K}\) and stored
\(W\in\mathbb{R}^{N\times K}\), \(Y=XW^\top\).

- **Static activation quantization:** \(s_X,z_X\) come from offline calibration. Runtime cost is low and scale reuse is high, but distribution shift can clip.
- **Dynamic per-tensor/per-token:** compute \(s_X\) from the current tensor or row. This adapts to outliers, but the reduction and cast read activation data and may dominate small decode GEMMs.
- **Weight qparams:** normally static, often per output channel.

The integer accumulator is

\[
C_{mn}=\sum_k(q^X_{mk}-z^X_m)(q^W_{nk}-z^W_n),
\qquad
\hat Y_{mn}=s^X_m s^W_n C_{mn}.
\]

For signed INT8 and \(K\) terms, the worst-case magnitude is on the order of
\(K\cdot127^2\). Check the actual code range and \(K\) before assuming INT32 cannot overflow. Symmetric weights remove weight zero-point corrections and simplify the kernel.

## SmoothQuant derivation

Let \(D=\operatorname{diag}(s_1,\ldots,s_K)\), with \(s_j>0\). Define

\[
X'=XD^{-1},\qquad W'=WD.
\]

Then

\[
X'{W'}^\top=XD^{-1}DW^\top=XW^\top.
\]

Thus activation channel \(j\) can be divided by \(s_j\) while the corresponding input channel of every consuming weight is multiplied by \(s_j\), without changing the floating-point function. SmoothQuant chooses scales using activation and weight maxima, commonly

\[
s_j=
\frac{\left(\max|X_{:,j}|\right)^\alpha}
{\left(\max|W_{:,j}|\right)^{1-\alpha}},
\quad 0\le\alpha\le1.
\]

Larger \(\alpha\) moves more difficulty from activations to weights. Epsilon handling is required for zero channels. The transformation must cross the correct graph boundary: scaling a residual branch, normalization output, or only one of several consumers incorrectly changes the model.

This equality holds before quantization. After separately quantizing \(X'\) and \(W'\), \(\alpha\) controls the error trade-off and must be calibrated.

## Kernel implications

An optimized W8A8 path fuses activation quantization, INT8 tensor-core GEMM, dequantization, bias, and sometimes residual/activation. A graph containing explicit Q/DQ nodes can still be fast if the compiler fuses them; an eager chain of reductions, casts, GEMM, and rescale may not be.

Prefill has large \(M\), so GEMM throughput often dominates. Decode has small \(M\), making scale computation, launches, and weight bandwidth important. Benchmark both. Check alignment of \(M,N,K\), supported per-token/per-channel combinations, accumulator/output dtype, and whether attention/LM-head layers remain high precision.

## Implementation ladder

### 1. Reference

Add `inference-lab/reference/int8_linear.py` with static and dynamic quantizers, explicit INT32 matmul, zero-point correction, and FP32 rescaling. Implement `smooth_linear(X,W,s)` and prove equality against FP64 before quantization.

### 2. Optimized

Vectorize per-token activation scales and per-channel weight scales. Fuse or compile the activation quantizer with GEMM where supported. Sweep \(\alpha\in\{0,0.25,0.5,0.75,1\}\) and compare static tensorwise with dynamic per-token activations.

### 3. Integrated

Calibrate a small decoder, fold smoothing into weights, serialize scales and recipe metadata, and load through one engine’s official W8A8/SmoothQuant route. Log the selected kernel. Reject silent BF16 fallback in the performance experiment.

## Three 3-hour sessions

1. **Math and oracle:** derive accumulator/rescale equations, implement static/dynamic paths, and test nonzero zero points.
2. **Smooth and calibrate:** collect per-channel maxima, apply the exact transform, sweep \(\alpha\), and evaluate layer output plus logits.
3. **Kernel and engine:** profile quantization, GEMM, and epilogue separately for decode/prefill shapes; trace checkpoint metadata to kernel dispatch.

Use the weekend to compare BF16, static W8A8, dynamic W8A8, and SmoothQuant on quality, model bytes, peak memory, TTFT, TPOT, and throughput.

## Experiments and deliverables

Produce:

- `reference/int8_linear.py`;
- `tests/test_int8_linear.py` with FP64 equivalence and overflow guards;
- `benchmarks/results/w8a8-shape-sweep.json`;
- `benchmarks/reports/smoothquant-alpha.md`;
- `notes/source_walks/w8a8-dispatch.md`;
- a pinned calibration manifest and converted-checkpoint checksum.

Use at least one outlier-channel synthetic case and one real activation capture. Warm up compiled kernels and report whether quantization time is included.

## Common failure modes

- Applying SmoothQuant scaling to the wrong axis or not to every weight consumer.
- Claiming equivalence after quantization rather than before it.
- Using calibration maxima from padding tokens or the wrong model mode.
- Comparing dynamic INT8 with a static path while excluding reduction overhead.
- Assuming every `int8` checkpoint uses INT8 activations.
- Missing saturation or INT32 overflow.
- Allowing an unsupported shape to fall back silently to BF16.

## Mastery gate

Given an activation-outlier plot and target GEMM shapes, choose static or dynamic scaling and an \(\alpha\) search. Derive the transformed tensors, implement a correct oracle, and provide profiler plus quality evidence that the deployed engine uses the intended W8A8 kernel.

## Canonical code paths and OSS bridge

Read the [SmoothQuant paper](https://arxiv.org/abs/2211.10438) with its [official repository](https://github.com/mit-han-lab/smoothquant). Inspect TensorRT-LLM’s [quantization example and format descriptions](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md), then follow the checkpoint loader into its quantized linear plugin. Compare TorchAO dynamic activation configurations. OSS opportunities include smoothing-axis tests, missing fallback warnings, calibration recipes, and decode-shaped benchmarks.

## Primary and authoritative sources

- [SmoothQuant: Accurate and Efficient PTQ for LLMs](https://arxiv.org/abs/2211.10438)
- [Official SmoothQuant implementation](https://github.com/mit-han-lab/smoothquant)
- [Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [TensorRT-LLM quantization examples](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md)
- [PyTorch quantization support](https://docs.pytorch.org/docs/stable/quantization-support)

## Next

Module 21 narrows activations back to FP16/BF16 and studies weight-only INT8/INT4 packing, dequantization, and decode-specific kernels.
