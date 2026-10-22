# Module 39 — Triton Kernels

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 20, paired with Module 40; three 3-hour sessions plus an abstraction and generated-code comparison  
**Prerequisites:** Modules 35–38; CUDA memory/execution, Tensor Core GEMM, tiling, pipelining, fusion, rooflines, numerical testing, and profiler discipline

## Why this matters

Triton lets a kernel author program blocks of values while its compiler maps those blocks to GPU threads, memory operations, and matrix instructions. It removes much indexing boilerplate but does not remove hardware constraints. Program shape, traversal order, masks, warps, stages, register pressure, target backend, and compiler version still decide correctness and speed.

For inference engineers, the central question is not whether Triton code is shorter than CUDA. It is whether a maintainable kernel covers the required shapes, generates suitable code on each target, integrates with dispatch and JIT caching, and beats a declared fallback. The learner’s working SM120 FlashInfer build supplies a concrete architecture and workload against which to test those claims.

## Outcomes

You can:

- map Triton program instances and blocked tensors to a CUDA grid and memory hierarchy;
- implement masked elementwise, reduction, and tiled matrix kernels;
- autotune only meaningful configurations and model specialization/cache cost;
- inspect TTIR, TTGIR, LLVM IR, PTX, cubin/SASS, and profiler evidence;
- separate source portability from numerical, binary, and performance portability.

## Mental model: programs own tiles

A Triton launch creates a grid of program instances. Each instance obtains coordinates with `tl.program_id`, constructs vector or multidimensional offsets, loads masked blocks, computes on them, and stores masked results. The compiler assigns elements of those blocks across participating threads. You choose the program grid, blocked algorithm, dataflow, and compile-time meta-parameters; the compiler chooses much of the lane-level mapping and lowering.

For \(C=A_{[M,K]}B_{[K,N]}\), a conventional grid has

\[
G_M=\left\lceil M/B_M\right\rceil,\qquad
G_N=\left\lceil N/B_N\right\rceil
\]

program tiles. Program \((p_m,p_n)\) accumulates \(C[p_mB_M:(p_m+1)B_M,\ p_nB_N:(p_n+1)B_N]\) while looping through \(K\) in \(B_K\) chunks:

\[
C_{\mathrm{tile}}\mathrel{+}=
A_{\mathrm{tile}}[:,k:k+B_K]\,
B_{\mathrm{tile}}[k:k+B_K,:].
\]

Use masks for every incomplete dimension and zero-fill masked \(K\) loads. Accumulate explicitly in FP32 when that is the intended contract, then apply the epilogue and cast once. Grouping consecutive \(M\)-tiles can improve L2 reuse, but the best order depends on shape and target.

`BLOCK_SIZE_*`, `num_warps`, and `num_stages` are not decorations. Larger blocks increase reuse and live values. More warps add cooperation but consume threads. More stages allow software-pipelined iterations but spend storage. `triton.autotune` can benchmark candidate `Config`s keyed by relevant dimensions, yet each key may trigger multiple executions and compilation. A broad search can corrupt in-place outputs, inflate cold latency, and create a large cache. Prune candidates by resource and shape reasoning first.

## Compilation, generated code, and portability

On NVIDIA, the useful inspection ladder is approximately

```text
Python + constexpr arguments
  → TTIR → TritonGPU IR (TTGIR) → LLVM IR → PTX → cubin/SASS
```

JIT identity includes source, compile-time values, target, and toolchain state. Separate first compilation, cache hit, autotuning, and warm execution. Use Triton’s official dump controls to retain each stage, then inspect the final binary with CUDA binary tools and correlate it with Nsight Compute. A `tl.dot` in source does not prove a particular Tensor Core instruction survived lowering.

Triton supports more than one backend, but portable source is not portable performance. Warp/wave size, shared memory, matrix instructions, supported dtypes, layout lowerings, and optimal configurations vary. Keep backend-specific configuration sets and correctness tolerances, and test generated code on every claimed target.

The RTX 5060 Ti is CC12.0/SM120 consumer Blackwell. B200 is CC10.0/SM100 datacenter Blackwell. A Triton result, autotune table, or PTX path from B200 is not local evidence. Reuse the successful FlashInfer environment only after verifying the Triton version and CUDA runtime are coherent; record the SM120 target and inspect what this compiler actually emitted.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/fused_linear.py` with PyTorch oracles for bias-SiLU, row reduction, and \(XW^\top+\text{bias}\). Cover noncontiguous strides, zero-sized dimensions, odd tails, NaN/Inf inputs, and explicit accumulation/output precision.

### 2. Optimized

Create `inference-lab/kernels/triton/fused_linear.py`. Begin with masked bias-SiLU, add a stable row reduction, then implement tiled GEMM with a fused epilogue. Use a small, justified autotune set for decode, batched decode, and prefill shapes. Dump compiler stages and record register/shared usage for the winners.

### 3. Integrated

Create `inference-lab/engine/triton_linear_backend.py` with device/dtype/layout/shape capability checks, target-aware cache keys, structured fallback reasons, and current-stream launch. Compare the custom kernel against PyTorch/cuBLAS and the selected vLLM/FlashInfer path; never force it across unsupported NVFP4 layouts.

## Three 3-hour sessions

1. **Language and correctness:** implement masked elementwise/reduction kernels; test tails, strides, precision, interpreter limitations, and Compute Sanitizer.
2. **Blocked GEMM:** implement `tl.dot`, grouping, fusion, and a bounded autotune; place winners on rooflines and inspect all generated-code stages.
3. **Integration and portability:** add dispatch/JIT manifests, exercise cold and warm paths on SM120, and perform a source-level portability audit without claiming untested hardware performance.

## Experiments and deliverables

Commit:

- `inference-lab/reference/fused_linear.py`;
- `inference-lab/kernels/triton/fused_linear.py`;
- `inference-lab/engine/triton_linear_backend.py`;
- `inference-lab/tests/test_triton_fused_linear.py`;
- `inference-lab/benchmarks/results/triton-sm120-shape-sweep.json`;
- `inference-lab/notes/generated-code/triton-sm120-pipeline.md`.

Record Triton/framework/compiler commits, backend and architecture, \(M,N,K\), strides, dtypes, meta-parameters, autotune key/winner, compile/cache/autotune time, warm latency, resource use, PTX/SASS evidence, error metrics, and fallback. Preserve raw dumps outside the result JSON and reference them by hash.

## Common failure modes

- Missing a load or store mask on one tail dimension.
- Using masked garbage rather than zero in a reduction or \(K\)-tail.
- Assuming contiguous input from a tensor’s logical shape.
- Letting autotuning repeatedly mutate an output or accumulator.
- Timing compilation/autotuning as kernel execution for one backend only.
- Specializing every scalar and causing cache explosion.
- Increasing blocks, warps, or stages until spills dominate.
- Accepting loose tolerances without identifying fast-math or accumulation changes.
- Treating `tl.dot` as proof of Tensor Core instructions.
- Calling one source “portable” after testing only SM100 or only SM120.

## Mastery gate

Given an odd \(M,N,K\), draw one program’s offsets and masks, predict its tile traffic, and identify its specialization key. Demonstrate reference parity, bounded autotuning, a warm speed result, and a complete TTIR-to-SASS trace on SM120. Explain one configuration reversal between decode and prefill and state exactly which portability claims remain untested.

## Canonical code areas and OSS bridge

Study Triton’s `python/triton/`, `lib/`, and NVIDIA backend beside the official tutorials. In vLLM, inspect `vllm/third_party/triton_kernels/`, `model_executor/layers/`, kernel registries, and JIT monitoring. Compare FlashInfer JIT/CUTLASS backends for the same operator contract.

Strong upstream work includes a minimal odd-shape reproducer, generated-IR regression, architecture guard, reduced autotune search, cache-key fix, or profiler-supported configuration. Submit source, oracle, environment, generated-code evidence, and cold/warm measurements.

## Primary and authoritative sources

- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton debugging guide](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- [Triton `Config` and autotuning API](https://triton-lang.org/main/python-api/generated/triton.Config.html)
- [Triton compiler source and dump controls](https://github.com/triton-lang/triton)
- [Triton language/compiler paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

## Next

Module 40 rebuilds the same reasoning with CUTLASS and CuTe, making layout algebra, hardware atoms, mainloops, epilogues, and architecture policies explicit.
