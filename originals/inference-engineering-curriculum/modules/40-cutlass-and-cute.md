# Module 40 — CUTLASS and CuTe

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 20, paired with Module 39; three 3-hour sessions plus a Triton–CUTLASS/CuTe comparison report  
**Prerequisites:** Modules 26, 28, and 35–39; packed and block-scaled formats, kernel contracts, SM targets, Tensor Core GEMM, pipelines, fusion, and generated-code inspection

## Why this matters

CUTLASS exposes production NVIDIA matrix-kernel machinery: copy and MMA atoms, hierarchical tiling, collective mainloops, epilogues, and schedulers. CuTe maps logical coordinates onto memory and threads. Because layout and hardware policy are explicit, a type-correct program can still describe the wrong transpose, scale layout, or architecture.

This applies directly to the learner’s working RTX 5060 Ti NVFP4 path. FlashInfer dispatches only after checking dtype, layout, shape, toolkit, and SM. Understanding that path makes backend selection auditable.

## Outcomes

You can:

- read and construct CuTe shapes, strides, layouts, tensors, tiles, and partitions;
- navigate CUTLASS device, kernel, collective, tiled-operation, and atom layers;
- assemble a mainloop, scheduler, and fused epilogue for a declared GEMM contract;
- compile, profile, verify, and inspect a deliberately small kernel set;
- distinguish reusable algorithm structure from architecture-specific atoms and binaries.

## Mental model: layouts compose; policies specialize

A CuTe layout maps a logical coordinate to a linear index. For a two-dimensional coordinate \((i,j)\),

\[
\mathcal{L}(i,j)=i\,d_0+j\,d_1.
\]

Row-major shape \((M,N)\) has stride \((N,1)\); column-major has \((1,M)\). Hierarchical modes can represent CTA, warp, instruction, or packed tiles. Composition and partitioning show which elements a thread group owns, whether access is contiguous, and whether a fragment matches its copy or MMA atom.

A CuTe `Tensor` combines an iterator or pointer with a layout. It does not magically transpose bytes. Two layouts can cover the same number of elements yet disagree about logical coordinates, packed nibbles, or NVFP4 scale blocks. Print and test small layouts before instantiating a large kernel.

CUTLASS 3.x organizes GEMM approximately as:

```text
device adapter
  → kernel = collective mainloop + collective epilogue + scheduler
      → tiled copy / tiled MMA
          → architecture-specific copy / MMA atoms
```

For \(D=\alpha AB+\beta C\), the mainloop moves and accumulates \(K\)-tiles; the epilogue reorganizes fragments for coalesced output and may fuse bias, activation, conversion, or scale generation. A scheduler maps tiles to CTAs and may be static, split-K, or persistent. Every generated kernel has a concrete tile, stage count, alignment, shared footprint, and target.

CuTe exists in C++ templates and NVIDIA’s Python DSL. The DSL lowers through MLIR and `ptxas`; C++ commonly produces AOT template instantiations. JIT adds cache and cold-start concerns; indiscriminate AOT creates enormous binaries.

## Portability and SM120

Logical tiling can often be recomposed with different atoms and policies. An atom, cubin, stage count, or result does not transfer automatically. `can_implement` must validate architecture, types, layouts, alignment, workspace, and shapes.

The RTX 5060 Ti is **SM120/CC12.0 consumer Blackwell**; B200 is **SM100/CC10.0 datacenter Blackwell**. NVIDIA documents SM100 architecture-conditional kernels as incompatible with RTX 50-series SM120. Use SM120 examples, compile the target required by the instruction, and inspect the binary. Do not borrow B200 clusters, Tensor Memory assumptions, or baselines.

Trace Python operator → vLLM registry → FlashInfer wrapper → `can_implement`/dispatch → generated or prebuilt kernel → SM120 launch. Record whether the path is CUTLASS C++, CuTe DSL, or another backend.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/cute_layout_oracle.py`. Model hierarchical shapes/strides, coordinate-to-index mapping, composition, tiling, and thread partitions in plain Python. Add golden row/column-major, transposed, padded, packed-FP4, and block-scale layouts with collision and coverage checks.

### 2. Optimized

Create `inference-lab/kernels/cutlass/sm120_gemm.cu` using the smallest suitable CUTLASS 3.x builder path, and `inference-lab/kernels/cute/sm120_layout_gemm.py` for a CuTe DSL counterpart where supported. Start with BF16/FP32 accumulation before attempting NVFP4. Limit instantiations to a few real \(M,N,K\) regimes, fuse one epilogue, run `can_implement`, and inspect PTX/SASS.

### 3. Integrated

Create `inference-lab/engine/cutlass_cute_backend.py` with versioned operand/layout descriptors, capability checks, workspace management, current-stream launch, target-aware JIT cache, and an explicit fallback. Compare its dispatch manifest with the working FlashInfer NVFP4 backend without replacing a proven production path merely to use a lower-level API.

## Three 3-hour sessions

1. **Layout algebra:** calculate and print layouts, tile and partition them over threads, then prove coverage, uniqueness, coalescing, and scale orientation on tiny examples.
2. **Kernel assembly:** connect atom → tiled operation → collective mainloop/epilogue → kernel adapter; compile only SM120 candidates and inspect resources and generated code.
3. **Integration and comparison:** add dispatch/cache/workspace handling, benchmark decode and prefill shapes, and compare Triton, CUTLASS/CuTe, cuBLAS, and FlashInfer on identical contracts.

## Experiments and deliverables

Commit:

- `inference-lab/reference/cute_layout_oracle.py`;
- `inference-lab/kernels/cutlass/sm120_gemm.cu`;
- `inference-lab/kernels/cute/sm120_layout_gemm.py`;
- `inference-lab/engine/cutlass_cute_backend.py`;
- `inference-lab/tests/test_cute_layout_and_dispatch.py`;
- `inference-lab/benchmarks/results/cutlass-cute-sm120.json`;
- `inference-lab/notes/source_walks/flashinfer-cutlass-cute-sm120.md`.

Record CUTLASS/CuTe/CUDA commits, build target and flags, logical and physical layouts, atom/instruction shape, CTA tile, stages, scheduler, epilogue, workspace, `can_implement` result, JIT/AOT state, binary targets, resources, correctness error, and warm/cold timings. Keep B200 documentation as contrast, never as a local baseline.

## Common failure modes

- Treating equal element counts as equivalent layouts.
- Swapping operands to repair a transpose while leaving scale orientation wrong.
- Hiding padding or packed-scale strides behind a format name.
- Instantiating thousands of kernels and mistaking build time for required complexity.
- Skipping `can_implement` and debugging the resulting launch failure.
- Using an architecture atom with a generic or incompatible target.
- Assuming a builder default is optimal for skinny decode \(M\).
- Comparing a fused CUTLASS epilogue with an unfused competitor.
- Mixing CuTe DSL cold JIT with warm AOT timings.
- Running or extrapolating an SM100/B200 kernel on SM120.

## Mastery gate

Given logical \(A,B,C\) and scale tensors, write each CuTe shape/stride mapping and show one thread partition by hand. Assemble a restricted kernel, justify atom/tile/stage/epilogue choices, pass `can_implement`, prove parity on tails, and identify the emitted SM120 binary/instruction path. Explain one Triton–CUTLASS/CuTe trade-off using maintainability, coverage, compile cost, and measured performance.

## Canonical code areas and OSS bridge

In CUTLASS, read `include/cute/`, `include/cutlass/gemm/`, collective builders, epilogues, profiler, and `examples/79_blackwell_geforce_gemm/`. In FlashInfer, inspect `flashinfer/gemm/`, `flashinfer/jit/`, generated sources, and SM120 FP4 templates. In vLLM, follow `model_executor/kernels/linear/`, quantization layout preparation, platform guards, and custom-op registration.

Useful contributions include a CuTe layout golden test, missing `can_implement` diagnostic, reduced SM120 instantiation, alignment/scale-layout fix, or measured kernel candidate. Pin versions because CUTLASS and CuTe DSL APIs evolve, and provide generated-code plus end-to-end dispatch evidence.

## Primary and authoritative sources

- [CUTLASS overview and profiler](https://docs.nvidia.com/cutlass/latest/overview.html)
- [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
- [CuTe layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html)
- [Getting started with CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html)
- [CuTe DSL overview and code generation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
- [CUTLASS Blackwell SM120 functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [NVIDIA CUTLASS source](https://github.com/NVIDIA/cutlass)

## Next

Module 41 packages a native kernel as a framework custom operator and follows `torch.compile` through dispatch, FakeTensor/meta behavior, graph guards, lowering, and caches.
