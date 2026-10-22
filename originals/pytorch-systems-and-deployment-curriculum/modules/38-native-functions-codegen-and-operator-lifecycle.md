# Module 38 — `native_functions.yaml`, Codegen, and the Operator Lifecycle

**Phase:** 6 — Native PyTorch and GPU integration  
**Time:** 12 focused hours  
**Prerequisites:** Modules 36–37, YAML/schema reading, dispatcher concepts, and C++ build literacy.

## Existing inference curriculum bridge

Connect [02 — Reproducible stack/toolchain](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), [37 — GEMM and Tensor Cores](../../inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md), [40 — CUTLASS/CuTe](../../inference-engineering-curriculum/modules/40-cutlass-and-cute.md), [41 — Custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [43 — Build/JIT/Ninja/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md), and [46 — vLLM contributions](../../inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md).

## Why this matters and outcomes

The fastest route to becoming useful in PyTorch is learning how a core operator is specified, generated, implemented, tested, and maintained. `native_functions.yaml` is central source-of-truth metadata, but editing it is a broad framework change, not the starting point for every local kernel. Operator lifecycle knowledge makes review feedback predictable and helps you choose an out-of-tree extension when that is the right engineering boundary.

You will trace an existing ATen op end to end, distinguish schema/codegen/meta/backend/autograd responsibilities, understand functional/in-place/out variants, and prepare a contribution-sized change with tests rather than a speculative YAML edit.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P38 — Native functions, codegen, and operator lifecycle**.

1. [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) — learn the API or system contract.
2. [PyTorch codegen source](https://github.com/pytorch/pytorch/tree/main/torchgen) — study the underlying design, paper, or specification.
3. [Custom ops tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch custom-ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch extension-cpp examples](https://github.com/pytorch/extension-cpp) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Core operator work begins with a schema: argument types/defaults, returns, alias annotations, tags, and variants. A family may have functional `add`, in-place `add_`, and `out=` variants. These are semantically related but not interchangeable: in-place updates version counters and aliases; `out=` has destination/storage requirements; functional returns new logical output. The schema supports dispatcher registration, generated declarations/wrappers, Python exposure, and transformation reasoning.

`aten/src/ATen/native/native_functions.yaml` supplies much of this declarative information. Code-generation tools consume it alongside related metadata to emit declarations, dispatch stubs, registration/wrapper glue, and other artifacts. Actual computation can be a structured native kernel, a composite implementation, a backend-specific function, or a generated stub calling a registered implementation. Meta/Fake behavior must infer output metadata without data; autograd may be expressed by derivatives definitions or composition. The exact generated artifacts and metadata files change, so trace one operator in the pinned source rather than memorizing an incomplete universal map.

For a typical lifecycle:

```text
schema + alias/mutation tags
  → generated API/dispatch glue
  → meta/shape contract + CPU/CUDA/composite kernels
  → autograd/functionalization/autocast behavior
  → Python/C++ API, export/compile consumers
  → tests, benchmarks, docs, deprecation/BC review
```

Adding an operator may require overload design, named dimensions or device rules, dtype promotion, error behavior, aliases, structured kernel metadata, CPU/CUDA/meta coverage, gradients, and compatibility policy. It is not enough to make a CUDA function compile. Many useful contributions instead improve an existing op’s test, meta behavior, error message, documentation, decomposition, or backend registration. For an experimental inference layout, ship an extension/TorchAO/vLLM kernel first; core ATen requires generality and maintenance commitment.

## Implementation ladder in `torch-lab/`

1. **Reference:** choose a small existing operator (e.g. a pointwise op) and create `torch-lab/aten_walks/<op>.md` tracing its schema, overloads, generated declarations, meta behavior, CPU/CUDA/composite path, derivative rule, and tests at a pinned commit.
2. **Optimized:** modify only a local clone/sandbox or an out-of-tree op to add an `out=`/functional variant and tests. Generate/build through the project’s normal tooling; inspect generated diff instead of hand-editing generated files.
3. **Integrated:** prepare a contribution proposal for a real issue: smallest code path, dispatch/autograd/meta matrix, BC considerations, tests, benchmark if performance-relevant, and alternatives. Do not submit an API without maintainer discussion.

## Three focused sessions

1. **Archaeology:** trace one op through YAML, source, generated output, dispatcher table, and Python call; make a source-link diagram.
2. **Semantic variants:** compare functional/in-place/out behavior with aliases, autograd, FakeTensor, and compiler transforms. Write failing tests for a deliberately wrong alias declaration.
3. **Contribution design:** select a narrowly scoped issue, build a test matrix, read contributor/BC guidance, and draft—not necessarily send—a review-ready plan.

## Experiments and deliverables

Commit `aten_walks/`, `operator_lifecycle_checklist.md`, source revision hashes, generated-artifact notes, `test_operator_variants.py`, and a mock PR description. The checklist must cover schema, aliases/mutation, dtype/device/layout, meta/fake, CPU/CUDA, gradients, autocast, functionalization, export/compile, performance, docs, and backwards compatibility.

## Common failure modes

- Hand-editing generated files and losing changes on the next codegen run.
- Adding a CUDA implementation while omitting schema/meta/error/dtype/CPU considerations.
- Treating functional, in-place, and `out=` as free aliases.
- Adding a narrowly hardware-specific format to core without an ownership/maintenance story.
- Benchmarking one kernel while altering a public operator contract.

## Mastery gate

Pick one existing ATen operator and explain its complete path from Python call to selected backend kernel and backward/meta behavior with pinned source links. Given a proposed new variant, identify every required contract/test area and justify whether it belongs in PyTorch core, TorchAO, or an external project.

## Primary sources, code paths, and OSS bridge

- [Dispatcher DevLog on `native_functions.yaml`](https://docs.pytorch.org/devlogs/dispatcher/2026-04-16-how-does-the-dispatcher-work/)
- [PyTorch contributor guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
- [PyTorch source: `native_functions.yaml`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
- [PyTorch C++ extension guidance](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [PyTorch developer discussion](https://dev-discuss.pytorch.org/)

Read `aten/src/ATen/native/native_functions.yaml`, `tools/codegen/`, `aten/src/ATen/native/`, `torchgen/`, `tools/autograd/`, and associated tests. Contributions that teach the lifecycle—small meta fixes, variant tests, correct error messages, decompositions, and codegen regressions—are excellent entry points. Follow issue/PR templates, run targeted and relevant full tests, and never edit generated outputs as the source of truth.

## Next

Module 39 moves from extension-level C++ to shipping an application against LibTorch and its ABI boundaries.
