# Module 34 — Triton with PyTorch

**Phase:** 5 — Capture, export, and compiler foundations  
**Time:** 15 focused hours  
**Prerequisites:** GPU memory/coalescing/occupancy, Modules 33 and 36–39 of the inference curriculum, and a PyTorch oracle.

## Existing inference curriculum bridge

Directly implement [35 — GPU architecture and Blackwell](../../inference-engineering-curriculum/modules/35-gpu-architecture-and-blackwell.md), [36 — CUDA execution/memory/streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), [37 — GEMM and Tensor Cores](../../inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md), [38 — Tiling/fusion](../../inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md), [39 — Triton kernels](../../inference-engineering-curriculum/modules/39-triton-kernels.md), and [40 — CUTLASS/CuTe](../../inference-engineering-curriculum/modules/40-cutlass-and-cute.md).

## Why this matters and outcomes

Triton lets you write GPU kernels in a Python-embedded DSL while retaining explicit control over program instances, tiles, layouts, masks, and compile-time specialization. It is ideal for learning and for many fused data-parallel kernels, but it is not a replacement for profiling, CUDA semantics, or vendor GEMM libraries.

You will write a correct tiled kernel, design a shape/layout capability boundary, benchmark it honestly against PyTorch, integrate it through a proper PyTorch operator boundary, and know when to stop and use Inductor/CUTLASS/cuBLAS instead.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P34 — Triton with PyTorch**.

1. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — learn the API or system contract.
2. [Triton language reference](https://triton-lang.org/main/python-api/triton.language.html) — study the underlying design, paper, or specification.
3. [Triton repository](https://github.com/triton-lang/triton) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

A Triton launch forms a grid of program instances. A vector of offsets may be built from `tl.program_id(axis)` and a compile-time block size; masked loads/stores protect tails. For rowwise `y_i=\operatorname{SiLU}(x_i+b_i)`, map programs to contiguous blocks to coalesce global memory. For a tiled matmul, a program computes a tile \(C_{m:n}\), loops over K tiles, and accumulates in a deliberately chosen wider type before output conversion.

Correctness requires more than matching one square, contiguous FP16 input. Define shape/rank/stride/dtype/device requirements, use masks on every out-of-bound access, choose explicit output/accumulation precision, handle zero dimensions, and compare NaN/Inf policy. Pointer arithmetic must honor strides; a kernel assuming contiguous rows needs an explicit fallback instead of silently indexing a transposed tensor wrong. Launch on the current PyTorch CUDA stream through a wrapper, not an assumed default stream.

Performance comes from memory coalescing, tile size, occupancy/register pressure, parallel reduction design, vectorization/layout, and reuse. `num_warps`, stages, and autotune configurations are hypotheses to measure per target/shape, not decorative constants. On your SM120 GPU, query capability at runtime and compile/cache only compatible variants. Do not set a global architecture variable to masquerade as hardware; an unsupported tensor-core instruction/layout must fall back safely.

Triton versions tied to PyTorch can change language/compiler details. Pin the package and preserve generated/cache artifacts when comparing. A custom Triton call that bypasses `torch.library` may work eagerly yet become opaque or invalid under `torch.compile`, FakeTensor, export, autograd, or CUDA graph capture. Module 35 supplies the integration contract.

## Implementation ladder in `torch-lab/`

1. **Reference:** implement `torch-lab/triton/reference_ops.py` in pure PyTorch: fused bias-SiLU, RMSNorm, and a reduction. Write property tests with noncontiguous and tail shapes before a kernel exists.
2. **Optimized:** add `torch-lab/triton/kernels.py` with one rowwise fused kernel and explicit capability checks. Benchmark a configuration sweep with fixed warmup/repetitions, profiler trace, and reference error. Implement tiled matmul only after the rowwise kernel is robust.
3. **Integrated:** wrap the selected kernel through Module 35’s custom-op contract; run eager/compiled/FakeTensor checks and use a dispatcher/capability fallback to PyTorch or a library path. Compare it to Inductor’s generated kernel and do not force it into NVFP4/FP8 layouts it does not implement.

## Three focused sessions

1. **Addressing:** draw mapping from program IDs to offsets, implement masked vector add/bias-SiLU, and test every tail/stride case.
2. **Reduction and tuning:** implement RMSNorm or a row reduction with stable accumulation; profile coalescing/launch count and sweep blocks/warps one variable at a time.
3. **Integration:** add custom-op metadata/tests, compile the surrounding model, inspect graph behavior, and make a source-linked comparison with eager/Inductor.

## Experiments and deliverables

Commit kernel source, `capabilities.py`, `test_triton_ops.py`, `bench_triton.py`, raw results, and `kernel_design.md`. Record Triton/PyTorch/CUDA/driver/GPU/SM, source hash, grid/meta-parameters, shapes/strides/dtypes, reference tolerances, generated artifact/cache key, launch count, cold compile, warm time, memory, and profiler evidence.

## Common failure modes

- Benchmarking only a shape divisible by every block size.
- Missing masks on a load while masking only the store.
- Comparing a fused kernel to several unfused PyTorch calls and calling the gap “Triton speed.”
- Tuning after losing output/gradient correctness or using a stale result cache.
- Pretending a pointer/low-bit layout is supported because it can be cast to a normal tensor.

## Mastery gate

Build a fused kernel with a precise capability matrix and exhaustive edge tests. It must safely fall back outside its contract, work on the current stream in a compiled path, and include a benchmark proving whether it beats the strongest fair PyTorch/Inductor baseline for the target workload.

## Primary sources, code paths, and OSS bridge

- [Triton language documentation](https://triton-lang.org/main/python-api/triton.language.html)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton repository](https://github.com/triton-lang/triton)
- [PyTorch compiler documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html)
- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

Read `python/triton/`, `lib/`, and PyTorch `torch/_inductor/codegen/triton*` only as pinned-source implementation details. Good contributions include correctness/tail tests, a minimized compiler regression, documentation of a target limitation, or a benchmark with reproducible artifacts. Start from tests and a reference, never from a copied “fast kernel” snippet.

## Next

Module 35 packages a Python or Triton implementation as a truthful PyTorch custom operator.
