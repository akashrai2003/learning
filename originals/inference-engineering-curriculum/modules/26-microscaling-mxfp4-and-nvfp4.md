# Module 26 — Microscaling, MXFP4, and NVFP4

**Phase:** 4 — Quantization  
**Pacing:** Week 13, paired with Module 25; three 3-hour sessions plus a weekend block-scaled GEMM audit  
**Prerequisites:** Modules 18, 21, and 25; grouped quantization, packed sub-byte data, FP8 scales, and kernel profiling

## Why this matters

Four-bit floating-point elements have too little range and precision to work well with one tensorwide scale. Microscaling divides tensors into small blocks, each with a shared scale. OCP MXFP4 and NVIDIA NVFP4 both use FP4 E2M1 values but differ in block size and scale representation. Those differences change accuracy, metadata, transpose handling, and hardware kernels.

## Outcomes

You can:

- decode FP4 E2M1 and shared-scale reconstruction;
- distinguish MXFP4 from NVFP4 numerically and physically;
- implement block quantization, scale packing, and shape validation;
- calculate effective bits/value including scales;
- trace a block-scaled checkpoint into a native Blackwell GEMM.

## Mental model: local exponent, tiny mantissa

For a block \(x_0,\ldots,x_{G-1}\), store one scale \(s_g\) and FP4 codes:

\[
q_i=\operatorname{cast}_{E2M1}(x_i/s_g),\qquad
\hat x_i=s_gq_i.
\]

E2M1 has one sign, two exponent, and one mantissa bit; finite reconstruction levels are sparse. Local scaling is therefore essential.

**MXFP4** follows the OCP microscaling specification: groups of 32 FP4 elements share an E8M0 scale, a power-of-two factor with no mantissa. Ideal payload is

\[
4+\frac{8}{32}=4.25\text{ bits/value}.
\]

Power-of-two scaling is cheap and standardized, but scale rounding itself loses precision.

**NVFP4** uses finer blocks of 16 E2M1 elements and a more precise E4M3 FP8 block scale, with a tensor-level scale in the double-quantized representation used by NVIDIA software. Its approximate local payload is

\[
4+\frac{8}{16}=4.5\text{ bits/value},
\]

plus amortized tensor-scale and alignment overhead. The finer block and richer scale can reduce error at a metadata cost.

Always check the consuming API: some interfaces expose already-composed FP16/FP32 dequant scales, while others store FP8 block scales plus a global scale. These are not byte-compatible merely because both say NVFP4.

## Shapes and scale orientation

For \(W[N,K]\) blocked along K:

\[
Q_W:[N,K/2]\ \text{bytes},\qquad
S_W:[N,\lceil K/G\rceil].
\]

For \(A[M,K]\), dynamic block scales are
\([M,\lceil K/G\rceil]\). Native block-scaled matrix multiplication requires exact K divisibility or a specified padding rule and specific physical scale layouts.

Microscaling is directional. Rowwise blocks of a tensor do not become valid columnwise blocks when bytes are transposed. Training kernels needing both \(X\) and \(X^\top\) should quantize each orientation from the high-precision source rather than transpose an already quantized tensor and compound error.

## Kernel implications

Native block-scaled tensor-core GEMM is architecture-specific. The kernel consumes packed FP4 operands and companion scales, accumulates in a wider type, and applies an epilogue. A fallback that expands FP4 to BF16 can save checkpoint memory but miss compute and bandwidth benefits.

Check GPU compute capability, CUDA/toolkit version, architecture flags, K/N alignment, supported scale layout, accumulation/output dtype, and whether the kernel accepts weight-only A16 or requires both operands quantized. Desktop and datacenter Blackwell variants are not interchangeable kernel targets.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/microscaling.py` with E2M1 value enumeration, MX E8M0 scale selection, NVFP4-style two-level scaling, nibble packing, and FP32 dequantization. Expose block axis and padding explicitly.

### 2. Optimized

Vectorize block amax/scale selection and compare fused block-scaled matmul with expand-then-matmul. If native hardware is unavailable, benchmark conversion separately and make no native-throughput claim.

### 3. Integrated

Export or obtain pinned MXFP4/NVFP4 checkpoints, inspect scale tensor shapes and metadata, and trace runtime kernel auto-selection. Run an architecture/shape support matrix and preserve logs proving whether CUTLASS, Transformer Engine, FlashInfer, or a fallback executed.

## Three 3-hour sessions

1. **Numerical format:** enumerate E2M1, implement MXFP4/NVFP4 emulators, and compare error on outlier blocks.
2. **Layout and GEMM:** pack operands/scales, test tails and transpose semantics, and estimate bytes/FLOPs for decode and prefill.
3. **Native integration:** trace checkpoint loader → quant method → kernel capability check → JIT/precompiled binary → launch.

The weekend report compares BF16, FP8, MXFP4, and NVFP4 for reconstruction, logits/PPL, effective storage, conversion overhead, and supported GEMM shapes.

## Experiments and deliverables

Commit:

- `reference/microscaling.py`;
- `tests/test_mxfp4_nvfp4.py` with golden scales/bytes;
- `benchmarks/results/fp4-block-shapes.json`;
- `benchmarks/reports/mxfp4-vs-nvfp4.md`;
- `notes/source_walks/block-scaled-kernel.md`.

Record block axis/size, element and scale formats, global-scale convention, rounding, padding, architecture target, toolkit, kernel library, and exact checkpoint revision.

## Common failure modes

- Calling MXFP4 and NVFP4 interchangeable.
- Omitting shared scales from memory estimates.
- Transposing quantized blocks instead of requantizing orientation.
- Blocking along an axis different from the kernel contract.
- Confusing an FP8 block scale with a composed FP16 dequant scale.
- Using a datacenter Blackwell cubin on a consumer Blackwell architecture.
- Forcing one global backend on a mixed FP8/NVFP4 checkpoint.
- Claiming native FP4 speed from an expand-to-BF16 fallback.

## Mastery gate

Given \(A[M,K]\), \(W[N,K]\), and a target architecture, produce exact packed and scale shapes for MXFP4 and NVFP4, calculate storage, implement golden dequantization, explain transpose handling, and prove which native or fallback kernel ran.

## Canonical code paths and OSS bridge

Read the [OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), TensorRT’s [quantized type equations and block sizes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html), Transformer Engine’s block-scaling recipes, and CUTLASS [block-scaled datatype/GEMM support](https://github.com/NVIDIA/cutlass). Then inspect FlashInfer or vLLM NVFP4 capability and `can_implement` checks. Valuable OSS work includes SM/shape support tests, scale-layout documentation, and correctness vectors shared across libraries.

## Primary and authoritative sources

- [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)
- [TensorRT quantization schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

## Next

Module 27 applies quantization to a stateful, growing activation: the KV cache, where scale axes and attention kernels differ from weight GEMM.
