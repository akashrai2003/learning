# Module 56 — AOTInductor Packaging and Native Deployment

**Phase:** 8 — Export and deployment  
**Time:** 15 focused hours  
**Prerequisites:** Modules 39–42, 51, and basic CMake/ELF/ABI debugging.

## Existing inference curriculum bridge

This continues [39 — Triton kernels](../../inference-engineering-curriculum/modules/39-triton-kernels.md), [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [42 — Compiler stack: Dynamo, Inductor, and Triton](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), [43 — Build/JIT/Ninja/CMake/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md), and [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). It turns compiler output into a deployable binary artifact rather than an opaque cache.

## Why this matters and outcomes

AOTInductor compiles an `ExportedProgram` into a packaged `.pt2` artifact intended for deployment outside a Python application, including C++ consumption. It is the PyTorch-native way to study the last mile from export constraints to generated native code, loader/runtime dependency, input contract, and operational test. It is not a promise that the package is an OS/GPU/version-independent executable.

You will be able to produce and audit a `.pt2` package, run it through Python and C++, preserve exact input shape/dtype/stride constraints, compare cold versus warm execution, and diagnose package/load failures with the build/ABI discipline from Module 40.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P56 — AOTInductor packaging and native deployment**.

1. [AOTInductor documentation](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) — learn the API or system contract.
2. [torch.export](https://docs.pytorch.org/docs/stable/export.html) — study the underlying design, paper, or specification.
3. [PyTorch C++ custom-op tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

The flow is `nn.Module → torch.export.ExportedProgram → AOTInductor compile/package → PT2 archive → Python or C++ package loader → generated kernels/runtime`. AOTInductor relies on export's soundness contract. Compilation fixes a target and compiler configuration; generated code, shared libraries, CPU ISA/CUDA target, and runtime dependencies are part of the artifact's provenance. The current official tutorial uses `torch._inductor.aoti_compile_and_package()` and `aoti_load_package()`; these interfaces are implementation-adjacent and need a pinned PyTorch version and fresh-process test.

The documented input contract is strict: inference inputs need compatible sizes, dtypes, and strides with export, subject to the declared dynamic dimensions. Turn on runtime input checks during development rather than waiting for an unfamiliar native crash. Package creation time, autotuning, first load, CUDA module/kernel initialization, and steady state answer distinct questions. A Python-free application still needs its declared native runtime/dependency contract; inspect dynamically loaded libraries and test in a clean environment.

For your SM120 machine, a CUDA package is a useful compiler/build lab if the installed PyTorch/Inductor path supports it. Start with CPU or a compact CUDA model and a minimal C++ runner; do not start by packaging the NVFP4 vLLM server. Torch-TensorRT can embed TensorRT-convertible regions in an AOTInductor package, but that is an optional later branch with its own Linux-only packaging/support constraints. Pin and test it independently.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/aoti/export_contract.py` and `test_eager_export.py`; reuse Module 51's artifact manifest and boundary input matrix.
2. **Optimized:** create `package_model.py` and compile one CPU or supported CUDA model to `.pt2`. Load it in a fresh Python process, enable development input checks, and test valid/invalid size/dtype/stride cases.
3. **Integrated:** add `cpp_runner/` instructions (not a generated binary in this curriculum) using the current AOTInductor CMake/C++ example. Write `native_deployment_dossier.md` with artifact contents/hashes, `ldd`/loader evidence, build/load/first/warm timing, and clean-environment results.

## Three focused sessions

1. **Artifact contract:** export/package a small model; compare eager, exported, and package output across declared shape buckets. Treat stride/layout as part of input validation.
2. **Native runner:** build a C++ executable against the exact LibTorch package, run it outside the build Python process, and diagnose one intentional missing-library or shape-contract error with evidence.
3. **Deployment decision:** compare JIT `torch.compile`, AOTInductor, ONNX Runtime, and TensorRT paths for a defined server target. State startup, portability, observability, custom-op, and version-risk tradeoffs rather than seeking one winner.

## Experiments and deliverables

Commit `export_contract.py`, `package_model.py`, `test_package_contract.py`, `cpp_runner/README.md`, `package_manifest.json`, and `native_deployment_dossier.md`. Record source/model/PyTorch/compiler versions, target CPU/GPU/SM, export constraints, Inductor settings, package hash/contents, native dependencies/resolved paths, input validation results, eager/export/package errors, build/load/first/warm timings, and packaging limitations.

## Common failure modes

- Calling a `.pt2` package a fully static executable with no dependency/version contract.
- Passing a shape-compatible but stride-incompatible input and blaming generated code.
- Timing AOT compilation or first CUDA initialization as steady inference latency.
- Reusing an artifact after changing PyTorch, driver, GPU target, or C++ ABI without validation.
- Folding Torch-TensorRT integration into the baseline before verifying plain AOTInductor.

## Mastery gate

Deliver a package that passes eager/export/Python-loader/C++-runner parity for its declared inputs in a clean process. From a failed load or bad input, classify the cause as contract, package, ABI/loader, compiler target, or numerical issue, then make the smallest safe correction.

## Primary sources, code paths, and OSS bridge

- [AOTInductor deployment tutorial](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html)
- [AOTInductor API reference](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)
- [Torch-TensorRT AOTInductor deployment](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/aot_inductor.html)
- [PyTorch Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor)

Read `torch/_inductor/`, `torch/csrc/inductor/aoti_runtime/`, package loader code, and matching tests at the pinned revision. Contributions should contain a reduced package/reload regression, explicit platform/version data, and tests on the supported target; do not report a package portability bug without clean-environment evidence.

## Next

Module 57 moves from server-native packages to the ExecuTorch edge runtime and EXIR lowering pipeline.
