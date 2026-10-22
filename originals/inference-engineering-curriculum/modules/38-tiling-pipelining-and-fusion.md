# Module 38 — Tiling, Pipelining, and Fusion

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 19, paired with Module 37; three 3-hour sessions plus the Capstone 3 design and correctness review  
**Prerequisites:** Modules 29–30 and 35–37; online algorithms, CUDA synchronization, memory hierarchy, GEMM shapes, Tensor Core evidence, occupancy, and rooflines

## Why this matters

Fast kernels are organized data movement. Tiling reuses values before eviction, pipelining overlaps movement with computation, and fusion avoids writing an intermediate only to read it again. Each technique also spends registers, shared memory, synchronization, code size, and specialization budget. A larger tile, deeper pipeline, or wider fusion boundary can therefore make a kernel slower or impossible to launch.

These choices are visible in the learner’s successful SM120 FlashInfer NVFP4 build: backend dispatch selects an architecture-, dtype-, layout-, and shape-specific mainloop plus epilogue, often through JIT. This module reconstructs those choices and measures them on the RTX 5060 Ti rather than inheriting a B200 schedule.

## Outcomes

You can:

- derive global-memory reuse from CTA, warp, instruction, and \(K\)-tile shapes;
- implement a correct tiled GEMM with tails and bank-conflict-aware shared layout;
- build and reason about a multistage producer–consumer pipeline;
- quantify when epilogue or operator fusion saves traffic;
- tune tiles, stages, and fusion using resources, rooflines, and real inference shapes.

## Mental model: retain, overlap, eliminate

Let a CTA compute a \(B_M\times B_N\) output tile and advance through \(K\) in chunks of \(B_K\). One mainloop stage loads \(B_MB_K\) elements of \(A\) and \(B_KB_N\) of \(B\), then performs \(2B_MB_NB_K\) FLOPs. Ignoring output traffic, its ideal stage intensity is

\[
I_{\mathrm{tile}}=
\frac{2B_MB_NB_K}
{s_A B_MB_K+s_B B_KB_N}
=\frac{2B_MB_N}{s_A B_M+s_B B_N}.
\]

Larger \(B_M,B_N\) increase reuse, but enlarge accumulator fragments and reduce grid parallelism. Larger \(B_K\) amortizes loop overhead but consumes shared memory and complicates tails. The CTA tile is partitioned into warp and MMA tiles; physical shared layouts may be swizzled or padded so concurrent accesses avoid bank conflicts and feed matrix instructions.

Without overlap, each stage costs roughly load latency plus compute time. A multistage circular buffer issues future global-to-shared copies while consumers compute on ready tiles. After prologue fill, ideal steady-state stage time approaches

\[
T_{\mathrm{stage}}\approx\max(T_{\mathrm{copy}},T_{\mathrm{compute}})
+T_{\mathrm{sync}},
\]

not their sum. This requires correct acquire/commit/wait/release ordering. More stages improve latency coverage only until shared-memory, register, barrier, or instruction overhead reduces residency or useful work. Prologue and drain costs matter for short \(K\).

Fusion removes materialization boundaries. For

\[
Y=\phi(\alpha AB+\beta C+b),
\]

a GEMM epilogue can apply scale, bias, activation, clamp, or quantization while accumulators are on chip, avoiding an output write/read pair. Fusion is most valuable when that traffic dominates. It can hurt when the fused operation increases live state, duplicates work, constrains a better library GEMM, creates too many variants, or forces unrelated consumers to wait. “One kernel” is not itself an objective.

## Scheduling and the local GPU

Tile order affects L2 reuse: nearby CTAs may share \(A\) or \(B\) panels. Split-\(K\) creates parallelism for skinny \(M,N\) but adds reduction traffic. Persistent schedules reduce launch/tile-assignment overhead but can load-balance poorly. Select policy by \(M,N,K\), not by model name.

On SM120, measure register allocation, shared memory, resident CTAs, bank conflicts, Tensor Core activity, and each memory level. The RTX 5060 Ti is CC12.0 consumer Blackwell; B200 is CC10.0 datacenter Blackwell. Different SM counts, memory, instruction contracts, and schedules make B200 stage counts and throughput non-transferable. The working FlashInfer build proves a local compilation/dispatch path; inspect its generated target and actual kernel before modeling it.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/tiling_model.py`. Simulate CTA coverage, masked tails, global tile loads, reuse, shared footprint, and pipeline prologue/steady-state/drain. Add an unfused PyTorch `linear → bias → SiLU` oracle and count its logical intermediate bytes.

### 2. Optimized

Create `inference-lab/kernels/cuda/tiled_fused_gemm.cu`. Progress from one-stage shared tiling to double buffering, then a configurable asynchronous pipeline. Add a fused bias-and-SiLU epilogue. Test padded and swizzled shared layouts, several \(B_M,B_N,B_K\), and unfused/fused variants against the oracle.

### 3. Integrated

Create `inference-lab/engine/tiled_linear_backend.py` with capability and shape guards, an explicit fallback, JIT/AOT cache identity, and structured dispatch logs. Integrate one layer-shaped benchmark, then compare against the selected FlashInfer/CUTLASS path without forcing a backend for unsupported shapes.

## Three 3-hour sessions

1. **Tile reuse:** derive traffic, implement one-stage shared GEMM, verify every tail, and profile coalescing, bank conflicts, and roofline movement.
2. **Pipeline:** add two then several stages, draw the producer–consumer state machine, and sweep \(K\), stage count, registers, shared memory, and occupancy.
3. **Fusion and dispatch:** fuse the epilogue, measure bytes and latency, integrate guards/cache keys, and compare decode, small batch, and prefill regimes.

## Experiments and deliverables

Commit:

- `inference-lab/reference/tiling_model.py`;
- `inference-lab/kernels/cuda/tiled_fused_gemm.cu`;
- `inference-lab/engine/tiled_linear_backend.py`;
- `inference-lab/tests/test_tiled_fused_gemm.py`;
- `inference-lab/benchmarks/results/sm120-tiling-pipeline-fusion.json`;
- `inference-lab/notes/profiles/gemm-pipeline-roofline.md`.

Record logical/padded shapes, tile hierarchy, shared layout, stages, synchronization protocol, epilogue, architecture target, cold/warm mode, registers, shared bytes, occupancy, bank conflicts, Tensor Core utilization, DRAM/L2 traffic, latency, and correctness error. Keep separate results for scalar, tiled, pipelined, fused, and production-library baselines.

## Common failure modes

- Loading a tile cooperatively, then reading it before all producers complete.
- Reusing a circular-buffer stage before consumers release it.
- Applying a full-tile store mask to \(M,N\) but forgetting the \(K\) tail.
- Increasing tiles or stages until spills erase reuse.
- Calling a padded shared layout “wasted” without measuring bank conflicts.
- Counting eliminated logical bytes but ignoring added conversion or scale traffic.
- Fusing across an operation needed independently or with incompatible numerics.
- Comparing cold JIT for one variant with warm cache for another.
- Autotuning only square GEMMs and deploying on skinny decode shapes.
- Copying SM100/B200 pipelines, clusters, or benchmark expectations to SM120.

## Mastery gate

For a supplied \(M,N,K\) and dtype, predict tile traffic, shared footprint, accumulator pressure, pipeline stages, and edge masks. Implement one-stage, multistage, and fused variants; prove correctness under sanitizer and odd shapes. Use profiler evidence to explain the winning configuration and one case where deeper pipelining or fusion loses.

## Canonical code areas and OSS bridge

Read CUTLASS collective mainloops, pipelines, schedulers, and epilogues beside `examples/79_blackwell_geforce_gemm/`. In FlashInfer, inspect `flashinfer/gemm/`, JIT cache construction, generated sources, and SM120 FP4 headers. In vLLM, follow `model_executor/kernels/linear/`, quantization preparation, and kernel configuration/dispatch.

Good contributions include a tail or synchronization test, a missing SM120 shape, a reduced JIT variant set, an epilogue correctness fix, or a profiler-backed tile configuration. Upstream evidence should include the reference, dispatch identity, generated target, cold/warm separation, and end-to-end layer effect.

## Primary and authoritative sources

- [CUDA Programming Guide: pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)
- [CUDA Programming Guide: asynchronous data copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [CUTLASS efficient GEMM and software pipelining](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
- [FlashInfer source](https://github.com/flashinfer-ai/flashinfer)

## Next

Module 39 expresses the same blocked computations in Triton, then audits what its compiler generated and which tuning decisions remain architecture-specific.
