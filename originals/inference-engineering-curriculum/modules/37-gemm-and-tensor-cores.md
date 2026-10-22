# Module 37 — GEMM and Tensor Cores

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 19, paired with Module 38; three 3-hour sessions plus a Capstone 3 kernel dossier  
**Prerequisites:** Modules 05, 18, 25–28, and 35–36; rooflines, quantized layouts, CUDA execution, memory hierarchy, occupancy, streams, and PTX/SASS inspection

## Why this matters

Transformer linears are matrix multiplications, but “GEMM uses Tensor Cores” is not a performance explanation. Decode may present only a few rows, prefill may present thousands, and quantized kernels must move packed values and scales before issuing matrix instructions. Shape, layout, precision, reuse, launch overhead, and epilogue work decide whether the Tensor Cores or a memory level is limiting.

The successful RTX 5060 Ti FlashInfer build is a valuable starting point: one SM120 backend compiled and dispatched. It does not establish Tensor Core utilization, instruction choice, or throughput for every shape. This module turns that success into evidence from arithmetic intensity, generated code, and profiler counters.

## Outcomes

You can:

- express inference linears as exact \(M,N,K\) GEMMs and identify decode versus prefill regimes;
- calculate FLOPs, compulsory bytes, arithmetic intensity, and a local roofline bound;
- explain scalar CUDA cores versus warp-cooperative Tensor Core MMA;
- distinguish input, product, accumulator, and output precision;
- prove which GEMM backend and native instruction executed on CC12.0.

## Mental model: one contraction, several ceilings

For row-major activations \(X[M,K]\) and weights \(W[N,K]\),

\[
Y[M,N]=\alpha XW^\top+\beta C,\qquad F\approx2MKN\text{ FLOPs}.
\]

With perfect reuse, a useful compulsory-traffic lower bound is

\[
Q_{\min}=s_XMK+s_WNK+s_YMN+Q_{\text{metadata}},
\qquad I_{\max}=F/Q_{\min}.
\]

Here \(Q_{\text{metadata}}\) includes packed scales, zero points, and padding. Real traffic is higher because tiles are reloaded, caches miss, tails are padded, activations are quantized, or intermediates are materialized. Measured performance obeys

\[
P_{\mathrm{achieved}}\le
\min(P_{\mathrm{local\ compute}}, I_{\mathrm{measured}}B_{\mathrm{level}}).
\]

Use measured clocks and bandwidth on the learner GPU, not a marketing peak. Hierarchical rooflines may reveal a shared/L1 or L2 ceiling before DRAM. During single-token decode, \(M\) is often one or a small live batch, so each weight has little reuse and the operation resembles GEMV. Larger prefill \(M\) amortizes weight reads and can become compute-bound.

A Tensor Core instruction computes a small matrix multiply-accumulate cooperatively across a warp or other architecture-defined thread group. A complete GEMM tiles the problem across CTAs, warps, instructions, and the \(K\)-loop. The hardware operation is not an arbitrary `float` triple loop: supported operand types, accumulator type, instruction shape, alignment, layout, and architecture target form an ABI. FP16 or BF16 inputs commonly accumulate in FP32; FP8 and block-scaled FP4 add conversion and scale semantics. Reduced input precision does not imply a four-bit accumulator.

Tensor Core eligibility is not Tensor Core efficiency. Too-small grids leave SMs idle; bad \(K\) alignment creates tails; poor layouts add transactions; large accumulator fragments consume registers; shared-memory tiles reduce resident CTAs. Occupancy helps hide latency but maximizing it can shrink useful tiles and reduce reuse. Inspect elapsed time, Tensor Core instruction activity, memory traffic, stalls, registers, shared memory, and achieved occupancy together.

## SM120 is its own result

The GeForce RTX 5060 Ti is consumer Blackwell **CC12.0/SM120**. B200 is datacenter Blackwell **CC10.0/SM100**. They do not share binary targets, memory systems, capacities, or every instruction/schedule. CUTLASS documents SM120 GeForce GEMMs separately. Never transfer a B200 tile configuration, TOPS figure, or benchmark percentage to this card.

For the working FlashInfer NVFP4 path, preserve the dispatch log, JIT target, packed/scale shapes, and cache artifact. Disassemble the loaded binary and correlate it with Nsight Compute. A kernel name containing “FP4” or “CUTLASS” is a hypothesis; generated instructions and counters are evidence.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/gemm.py` with an FP64 triple-loop oracle and vectorized PyTorch reference. Accept explicit strides, transpose flags, \(\alpha,\beta\), bias, and tails. Add a shape reporter for model linears that prints \(M,N,K\), logical bytes, and ideal arithmetic intensity.

### 2. Optimized

Create `inference-lab/kernels/cuda/tensor_core_probe.cu`. Compare a scalar CUDA GEMM, cuBLAS/cuBLASLt, and one minimal supported warp-level MMA path. Accumulate against FP32, guard edges, and report kernel resources. Sweep real decode and prefill shapes rather than only square matrices.

### 3. Integrated

Create `inference-lab/engine/gemm_dispatch.py` to select a declared backend by device, dtype, layout, alignment, and \(M,N,K\). Trace one Gemma 4 FP8 linear and one NVFP4 linear through vLLM → FlashInfer/CUTLASS → JIT or prebuilt binary. Emit structured selection and rejection reasons.

## Three 3-hour sessions

1. **Shapes and ceilings:** inventory model linears, derive FLOPs/bytes, benchmark cuBLAS across \(M\), and place points on local rooflines.
2. **Instruction proof:** implement scalar and MMA probes, inspect PTX/SASS, and relate instruction mix to alignment, tails, resources, and numerical error.
3. **Engine trace:** capture FP8/NVFP4 backend selection, cold JIT and warm launches, then compare microkernel, layer, and serving effects.

## Experiments and deliverables

Commit:

- `inference-lab/reference/gemm.py`;
- `inference-lab/kernels/cuda/tensor_core_probe.cu`;
- `inference-lab/engine/gemm_dispatch.py`;
- `inference-lab/tests/test_gemm_contract.py`;
- `inference-lab/benchmarks/results/sm120-gemm-regimes.json`;
- `inference-lab/notes/source_walks/flashinfer-nvfp4-gemm.md`.

Record logical and padded shapes, strides, operand/scale/accumulator/output types, backend, target architecture, compiler/library commits, cold/warm status, FLOPs, bytes by memory level, latency, TFLOP/s, effective bandwidth, resource usage, and profiler command. Include \(M=1\), small continuous batches, and representative prefills.

## Common failure modes

- Swapping \(W[N,K]\) and \(W^\top[K,N]\) while keeping plausible shapes.
- Counting one fused multiply-add as one FLOP in one result and two in another.
- Omitting quantization scales, padding, or conversion traffic.
- Assuming a library call necessarily selected Tensor Cores.
- Equating Tensor Core activity with proximity to the relevant roofline.
- Benchmarking only large square GEMMs for an autoregressive workload.
- Maximizing occupancy while causing smaller tiles or extra global traffic.
- Mixing cold JIT/autotuning with steady-state kernel latency.
- Treating SM100/B200 results or cubins as SM120 evidence.

## Mastery gate

Given three model linears at decode and prefill \(M\), derive exact shapes, FLOPs, compulsory bytes, and roofline classification. Predict which case cannot fill the GPU, then verify backend, native instruction family, resources, and measured limiter on the RTX 5060 Ti. Explain any prediction error without citing peak TOPS alone.

## Canonical code areas and OSS bridge

Trace `vllm/model_executor/layers/linear.py`, `vllm/model_executor/kernels/linear/`, quantization methods, platform capability checks, and custom-op registration. In FlashInfer, follow `flashinfer/gemm/`, JIT generation, generated sources, and SM120 headers. In CUTLASS, connect device GEMM, collective mainloop, tiled MMA, and epilogue.

A useful upstream contribution is an SM120 shape regression, clearer `can_implement` reason, missing alignment test, or benchmark with dispatch and generated-code proof. Compare against vendor libraries and preserve correctness; do not submit an isolated speed percentage.

## Primary and authoritative sources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [PTX ISA: warp-level matrix multiply-accumulate](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [cuBLAS documentation: GEMM and Tensor Core usage](https://docs.nvidia.com/cuda/cublas/)
- [CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [CUTLASS Blackwell functionality, including SM120](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [Nsight Compute roofline and profiling documentation](https://docs.nvidia.com/nsight-compute/)

## Next

Module 38 opens the GEMM and deliberately changes tile reuse, pipeline depth, scheduling, and fusion to move its roofline position.
