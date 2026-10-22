# Module 05 — Performance Modeling and Roofline

> **Phase:** 1 — Foundations  
> **Time:** Week 3; two 3-hour sessions plus a 2–4 hour benchmark memo  
> **Prerequisites:** Modules 03–04, dimensional analysis, and basic GPU timing

## Why this matters

Optimization without a bound is guesswork. A kernel that reaches only 10% of peak FLOP/s may be excellent if memory bandwidth is its real ceiling; a 2× faster kernel may not move request latency if launches or queueing dominate. Roofline analysis converts tensor shapes and dtypes into a falsifiable bottleneck hypothesis before you write a kernel.

## Outcomes

You can:

- calculate FLOPs, compulsory bytes, and arithmetic intensity for core inference operations;
- place prefill and decode on different performance regimes;
- construct compute-, memory-, launch-, and capacity-bound lower bounds;
- measure effective bandwidth and achieved FLOP/s correctly;
- explain when fusion, batching, quantization, or tiling can move a workload.

## Mental model: every operation pays in work, movement, and launches

The classic roofline says achievable performance is bounded by:

\[
P \le \min(P_{\text{peak}},\ I\,BW_{\text{peak}}),
\qquad I=\frac{\text{operations}}{\text{bytes transferred}}.
\]

It is an upper bound, not a prediction. Real kernels face lower ceilings from memory coalescing, cache hierarchy, occupancy, instruction mix, synchronization, and launch overhead. For short decode steps, a latency model is often clearer:

\[
t \gtrsim \max\left(\frac{F}{P_{\text{usable}}},
\frac{Q}{BW_{\text{usable}}}\right)+N_{\text{launch}}t_{\text{launch}}.
\]

Start with compulsory HBM bytes; refine with profiler-measured traffic. Never add cache reuse merely because it would be convenient.

## Theory, shapes, and inference regimes

For \(C:[M,N]=A:[M,K]B:[K,N]\), dense GEMM performs roughly \(2MKN\) FLOPs. A simple byte floor is the inputs read once and output written once:

\[
Q_{\min}=p_A MK+p_B KN+p_C MN,
\]

where \(p\) is bytes per element. Actual traffic can be higher; efficient tiling tries to approach this floor.

A decode projection has \(X:[B,D]\), \(W:[D,F]\), and \(Y:[B,F]\):

\[
F_{\text{ops}}=2BDF.
\]

If each BF16 weight is streamed once, weight arithmetic intensity is approximately \(2B/2=B\) FLOP/byte before other traffic. At \(B=1\), it is strongly bandwidth-oriented. A 4-bit weight can raise this idealized intensity, but only if scales and packed payload are consumed directly by the matrix kernel.

During prefill, flatten `[B,S,D]` to `[T,D]`, \(T=BS\). Many token rows reuse one weight matrix, increasing intensity and exposing larger GEMMs. This explains why prefill and decode need separate benchmark sweeps.

For full causal attention, score and value contractions scale approximately as \(4BHS^2D_h\) FLOPs. A naive implementation also writes `[B,H,S,S]`. FlashAttention preserves exact attention math while reducing HBM traffic by tiling and online softmax. Its gain is an IO argument, not a reduction in the dense attention FLOP count.

Decode must additionally read historical KV. For \(L\) layers and \(H_{kv}\) KV heads, cache payload is:

\[
Q_{KV/token}\approx 2LBSH_{kv}D_h\,p_{KV}.
\]

That term grows with context length and can dominate even after weight quantization.

## Implementation ladder

1. Add `benchmarks/cost_model.py` with pure functions for GEMM, attention, weight, activation, and KV FLOP/byte estimates.
2. Make dtype bytes, scale metadata, padding, batch, prompt length, decode length, and cache reuse explicit inputs.
3. Add `benchmarks/microbench.py` using warmups, repeated trials, CUDA events, synchronization, and quantiles.
4. Sweep linear shapes representative of prefill (`M=T`) and decode (`M=B`) rather than square GEMMs only.
5. Measure copy bandwidth, GEMM achieved FLOP/s, and elementwise effective bandwidth. Compare measured points with device ceilings in `notes/roofline.csv`.
6. Add `tests/test_cost_model.py` with hand-derived cases and unit checks. A byte/GiB or ms/s error should fail loudly.

## Two 3-hour sessions

**Session A — model the ceilings**

- **0:00–0:35 — Derivation:** calculate work and bytes for linear, norm, attention, and KV reads.
- **0:35–1:05 — Regime comparison:** place prefill and batch-1 decode on a paper roofline.
- **1:05–1:15 — Break.**
- **1:15–1:55 — Harness lab:** implement synchronized, warm, quantile-based measurement.
- **1:55–2:30 — Sweep:** vary \(M\), dtype, and layout; calculate achieved bandwidth/FLOP/s.
- **2:30–3:00 — Profiler bridge:** compare model bytes with Nsight Compute DRAM and cache metrics.

**Session B — test the model**

- **0:00–0:45:** collect prefill/decode sweeps at multiple batches and contexts.
- **0:45–1:30:** build empirical compute and bandwidth ceilings.
- **1:30–2:15:** profile outliers and refine compulsory versus actual bytes.
- **2:15–3:00:** defend one optimization and one predicted non-improvement in the memo.

## Experiments and deliverables

Submit the cost model, tests, raw benchmark CSV, roofline plot, and a one-page bottleneck memo. For one decode projection and one prefill projection, state:

- predicted dominant ceiling;
- ideal and profiler-observed bytes;
- achieved fraction of the relevant ceiling;
- one optimization that should help and one that should not.

Then double batch size. Explain how weight reuse, latency, and aggregate throughput change. Repeat at two context lengths to expose KV traffic.

## Failure modes

- Using advertised sparse or low-precision peak for a kernel that does not issue those instructions.
- Counting one fused multiply-add inconsistently as one versus two FLOPs.
- Counting tensor size but not reads plus writes.
- Using allocation size as a substitute for memory traffic.
- Benchmarking square GEMMs instead of inference shapes.
- Timing JIT compilation, allocator growth, or asynchronous launch submission.
- Claiming a kernel is compute-bound from GPU utilization alone.
- Applying a kernel-level speedup directly to end-to-end latency.

## Mastery gate

Given shapes, dtypes, and device ceilings, calculate arithmetic intensity and identify the first roofline bottleneck. Your measured result must include raw samples, p50/p95, synchronization method, warm/cold status, and a model-versus-profiler byte comparison within a reasoned tolerance.

## Primary sources

- [Roofline: An Insightful Visual Performance Model](https://digicoll.lib.berkeley.edu/record/136692/files/EECS-2008-134.pdf)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Compute roofline documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

## Code paths to read

- Triton: `python/tutorials/03-matrix-multiplication.py`
- Triton: `python/tutorials/06-fused-attention.py`
- FlashAttention: `csrc/flash_attn/src/`
- Your lab: `benchmarks/cost_model.py`, `benchmarks/microbench.py`

## OSS bridge

Pick one vLLM profiler trace and classify its largest kernels by shape and regime. Use Nsight metrics to refine, not replace, your analytic model. Add measured ceilings as a device profile so later modules can predict whether an architectural change shifts FLOPs, bytes, or launch count.

## Next

Module 06 composes the operations into a complete decoder forward pass and attaches this cost vocabulary to real model boundaries.
