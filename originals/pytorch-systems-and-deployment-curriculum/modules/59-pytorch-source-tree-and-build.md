# Module 59 — PyTorch Source Tree and Build

**Phase:** 9 — PyTorch source and contribution engineering  
**Time:** 18 focused hours  
**Prerequisites:** Modules 35–42, 43, and 56; C++/CUDA build basics; sufficient host disk/RAM for a separate source checkout.

## Existing inference curriculum bridge

This is the upstream counterpart to [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [42 — Compiler stack: Dynamo, Inductor, and Triton](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), [43 — Build/JIT/Ninja/CMake/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md), and [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Your FlashInfer SM120 compile postmortem becomes an advantage here: PyTorch source builds have the same resource, architecture, ABI, cache, and incremental-build realities on a larger scale.

## Why this matters and outcomes

Contributing effectively requires a navigable mental map of PyTorch, not a monolithic checkout. Python front-end code, dispatcher registrations, ATen operator definitions, generated bindings, autograd, distributed C10D, compiler code, tests, CMake, third-party libraries, and CI conventions sit at different layers. A source build gives you executable evidence of which layer a behavior belongs to—but an indiscriminate full CUDA rebuild is a poor first move.

You will be able to choose a narrow source walk, create a separate reproducible development environment, make an editable source build with a controlled resource budget, locate an operator/compiler/distributed feature from Python to tests, and preserve a build manifest suitable for a future issue or PR.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P59 — PyTorch source tree and build**.

1. [PyTorch from-source build](https://github.com/pytorch/pytorch#from-source) — learn the API or system contract.
2. [PyTorch repository](https://github.com/pytorch/pytorch) — study the underlying design, paper, or specification.
3. [PyTorch developer wiki](https://github.com/pytorch/pytorch/wiki) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch developer notes](https://docs.pytorch.org/docs/stable/community/contribution_guide.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch source repository](https://github.com/pytorch/pytorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch developer discussion](https://dev-discuss.pytorch.org/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

At a high level, `torch/` is the Python package/front end; `torch/csrc/` contains bindings and substantial C++ implementation; `aten/` is the tensor library; `aten/src/ATen/native/` contains modern operator implementations with CPU/CUDA/backend subdirectories; `c10/` contains core portable components; `torchgen/` and `aten/src/ATen/native/native_functions.yaml` drive generated operator plumbing; `torch/distributed/` and `torch/csrc/distributed/` cover distributed APIs/C10D; `torch/_dynamo`, `torch/_functorch`, and `torch/_inductor` cover major compiler portions; `test/` and `test/cpp/` define behavioral evidence.

Start at the public API call, use `rg` to find its schema/registration/test, and draw the smallest dispatch/source map. Do not edit generated files until you understand their generator and source-of-truth. A Python-only change may be visible through an editable install; a C++/CUDA/header/schema change needs the relevant native rebuild and may change ABI. Pin commit, submodules, Python/compiler/CMake/Ninja/CUDA/driver, selected build options, architecture target, and environment. The upstream instructions currently use `spin` for common development commands and document editable build/test workflows; follow the pinned checkout's own `CONTRIBUTING.md`, because source instructions change.

Your 5060 Ti supports a focused CUDA source walk, not a license to build every optional backend blindly. Put the checkout and build cache outside the working vLLM environment, preserve the known-good wheel environment, and begin CPU/Python or selected test targets. Use bounded build parallelism based on host memory; your prior OOM lesson applies: outer Ninja concurrency and inner compiler work multiply. Save the failure's command/log/build graph before cleaning it.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/pytorch-source/source_map.md` for one operation such as `aten::add`, `scaled_dot_product_attention`, or a small custom-op path. Trace API → schema → generated/dispatcher route → backend implementation → tests.
2. **Optimized:** write `build_plan.md` for a separate checkout/environment: disk/RAM budget, branch/commit/submodules, Python/toolchain, CPU/CUDA target, exact build command, cache locations, `MAX_JOBS` policy, smoke import, and rollback/cleanup scope. Do not apply it to the vLLM environment.
3. **Integrated:** perform only the smallest justified editable/source build and `source_build_manifest.json`. Change a harmless local Python/doc/test probe, rebuild the necessary target, verify `torch.__file__` points to the checkout, and run one focused test.

## Three focused sessions

1. **Source cartography:** follow one eager op and one compiler/export operation through code/tests. Compare the source map with an actual dispatcher/graph dump from earlier modules.
2. **Build forensic setup:** inspect upstream build instructions, toolchain/ABI constraints, CMake/Ninja output, and cache locations. Establish host-memory limits and a deliberately narrow rebuild target.
3. **Incremental proof:** make a reversible local probe, verify which files trigger Python-only versus native rebuilds, then remove or preserve the probe in a clean branch without touching user projects.

## Experiments and deliverables

Commit `source_map.md`, `build_plan.md`, `source_build_manifest.json`, `build_log_excerpt.md`, and `focused_test_evidence.md`. Capture source remote/commit/submodules, Python/PyTorch/toolchain/CUDA/driver/SM, CMake/Ninja/ccache settings, resource limits, build flags, selected targets, elapsed/peak host memory, import location, test command/result, generated-file provenance, and any deferred CUDA/full-build rationale.

## Common failure modes

- Building source inside or over the environment that runs a known-good server.
- Editing a generated binding instead of its schema/generator input.
- Assuming a driver CUDA capability string selects the compiler toolkit or SM target.
- Turning a host OOM into repeated full rebuilds without reducing concurrency.
- Reporting a source-build issue without commit, compiler, command, and focused reproducer.

## Mastery gate

Given an API behavior, produce a source/test map that another contributor can follow. Build the minimal required target in an isolated environment, prove which `torch` is imported, and explain the exact host-resource/ABI/architecture controls used. Classify one build failure from its log before proposing any change.

## Primary sources, code paths, and OSS bridge

- [PyTorch `CONTRIBUTING.md`](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
- [PyTorch source repository](https://github.com/pytorch/pytorch)
- [PyTorch contribution guide](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions)
- [PyTorch C++ extension documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html)

Read the current checkout's `CONTRIBUTING.md`, `README.md`, `aten/src/ATen/native/native_functions.yaml`, `torchgen/`, `CMakeLists.txt`, `torch/csrc/`, and the relevant `test/` files. First source contributions are often documentation, test, diagnostic, or narrowly scoped Python issues; source mastery comes from tracing evidence, not from forcing a giant CUDA build.

## Next

Module 60 teaches the test, CI, and triage work that turns a local observation into reviewable evidence.
