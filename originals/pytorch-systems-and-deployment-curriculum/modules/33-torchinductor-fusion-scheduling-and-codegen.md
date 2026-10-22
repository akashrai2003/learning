# Module 33 — TorchInductor: Fusion, Scheduling, and Code Generation

**Phase:** 5 — Capture, export, and compiler foundations  
**Time:** 12 focused hours  
**Prerequisites:** Modules 29–32, GPU hierarchy/roofline intuition, and profiler literacy.

## Existing inference curriculum bridge

This is the framework-level counterpart of [05 — Performance modeling and roofline](../../inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md), [36 — CUDA execution/memory/streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), [37 — GEMM/Tensor Cores](../../inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md), [38 — Tiling/fusion](../../inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md), [39 — Triton kernels](../../inference-engineering-curriculum/modules/39-triton-kernels.md), and [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md).

## Why this matters and outcomes

TorchInductor is the default backend behind `torch.compile`; it lowers captured graphs into optimized code across targets. Its gains usually come from fewer launches, less intermediate memory traffic, better layouts, and specialized kernels—not from a mysterious faster PyTorch. Reading its artifacts makes you a sharper performance engineer and a more useful upstream contributor.

You will be able to relate an FX-level expression to generated kernels, identify legal fusion opportunities and barriers, inspect code/cache artifacts without treating them as stable APIs, and make an evidence-based decision between Inductor-generated code, a library template, or a bespoke kernel.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P33 — TorchInductor fusion, scheduling, and codegen**.

1. [TorchInductor docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) — learn the API or system contract.
2. [Inductor CPU backend paper](https://arxiv.org/abs/2301.13448) — study the underlying design, paper, or specification.
3. [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Given a functional graph, Inductor applies decompositions/lowerings, constructs an internal scheduling representation, chooses layouts and fusion groups, then emits target code/wrappers. GPU paths often use Triton for pointwise/reduction-like kernels and specialized/template/library paths for some matrix operations; CPU paths may emit C++/vectorized code. The exact internal IR and filenames evolve, so learn concepts and inspect the pinned source/artifacts rather than building tooling against an undocumented class.

Fusion avoids materializing an intermediate. If `a = x * scale; y = silu(a + bias)`, separate kernels write/read `a` through HBM; a fused kernel can keep it in registers or shared/local memory. But fusion changes scheduling/register pressure and may reduce occupancy or block use of a better library GEMM. Reduction ordering changes floating-point results; layout conversion can dominate a seemingly cheap op. Thus a graph with fewer kernels is not automatically faster or numerically identical bit-for-bit.

For an elementwise chain with \(n\) elements, arithmetic intensity is low and HBM traffic dominates; fusion helps most by reducing reads/writes. For GEMM, \(2MNK\) FLOPs and reusable tiles may be compute-bound; Inductor may select an autotuned template or leave a library call rather than synthesize a naive loop. Inspect profiles: kernel name, launch count, shapes, achieved time, allocation, and source. Never infer a tensor-core path from a high-level `matmul` alone.

Generated code belongs in a cache keyed by versions/options/shape/device assumptions. It can include host compilation, autotuning, and JIT work, so separate compilation overhead from warm performance. Preserve an artifact copy and metadata for debugging; delete only with a reason. Module 40 extends this to C++/CUDA/Ninja ABI mechanics.

## Implementation ladder in `torch-lab/`

1. **Reference:** make `torch-lab/inductor/reference_blocks.py` with unfused PyTorch expressions for bias+activation+residual, layer norm, and a small matmul epilogue. Test outputs/gradients over varied shapes and strides.
2. **Optimized:** compile one block at a time with trace/debug artifacts enabled for the pinned release. Count generated launches/intermediates, inspect emitted source, and compare eager, compiled default, and one documented option/mode.
3. **Integrated:** use a bounded decoder prefill or vision block with shape policy from Module 31. Correlate Inductor artifacts with profiler timelines, cache state, CUDA graph behavior, and end-to-end quality/TTFT/TPOT—not a synthetic microbenchmark alone.

## Three focused sessions

1. **Lowering walk:** follow a simple expression from FX graph through generated source/wrapper and profiler kernel timeline. Label every materialized intermediate and launch.
2. **Fusion experiment:** add/remove a fusion-friendly chain and a fusion barrier (view/layout, custom op, reduction). Measure memory traffic proxy, launches, registers/occupancy evidence where available, numeric drift, and warm time.
3. **Matmul choice:** sweep M/N/K and layout, determine when generated versus library/template code is selected, and build a decision note with source artifacts.

## Experiments and deliverables

Commit `inductor_cases.yaml`, preserved trace/source artifacts or hashes, `kernel_inventory.csv`, `test_inductor_parity.py`, and `fusion_decisions.md`. Include commit/version, cache location/state, backend options, shape/stride/dtype, graph hash, generated source hash, launch count, compile/autotune time, peak memory, p50/p99 warm time, numerical error, and profiler evidence.

## Common failure modes

- Reading generated code from a different cached configuration than the timed process.
- Claiming fusion based on FX appearance without a profiler trace.
- Changing shape, layout, mode, and dtype in a single comparison.
- Treating `max-autotune` or any mode as a universal win without startup/memory accounting.
- Replacing a correct vendor matmul with a custom kernel before measuring its true bottleneck.

## Mastery gate

Given a compiled block, trace one output back to its graph and each generated launch. Identify one safe fusion opportunity and one reason not to fuse, then demonstrate an end-to-end improvement that preserves output/gradient contracts and states cold versus warm costs.

## Primary sources, code paths, and OSS bridge

- [TorchInductor GPU profiling guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_inductor_profiling.html)
- [`torch.compile` options and modes](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch 2 paper and TorchInductor tutorial](https://pytorch.org/blog/pytorch-pytorch-2-paper-tutorial/)
- [PyTorch compiler documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html)
- [PyTorch source](https://github.com/pytorch/pytorch)

Read `torch/_inductor/graph.py`, `lowering.py`, `scheduler.py`, `codegen/`, `select_algorithm.py`, `codecache.py`, and tests tied to your issue. Contributions should start with a minimal graph plus eager/compiled correctness and profiler evidence; useful first work includes a decomposition regression, a fusion correctness test, diagnostics, or a cache-key bug.

## Next

Module 34 gives you direct control over the Triton kernels that Inductor often emits.
