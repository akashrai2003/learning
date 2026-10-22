# Module 30 — `torch.compile` and TorchDynamo

**Phase:** 5 — Capture, export, and compiler foundations  
**Time:** 12 focused hours  
**Prerequisites:** Module 29, Python execution model, and profiler basics.

## Existing inference curriculum bridge

Use [34 — Continuous batching and chunked prefill](../../inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md), [36 — CUDA execution/memory/streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), and [43 — Build/JIT/Ninja/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md).

## Why this matters and outcomes

`torch.compile` is a runtime compilation interface, not one backend or one fixed optimization level. TorchDynamo observes Python frames, extracts graphable tensor regions, emits FX graphs to a backend, caches results behind guards, and falls back to Python where needed. Understanding the boundary prevents both superstition (“compile is broken”) and unsafe workarounds (“just allow it in graph”).

You will be able to run controlled eager/compiled comparisons, find graph breaks, choose a compile mode only after measurement, distinguish capture/compile/warm execution time, and write minimal reproductions appropriate for PyTorch or vLLM contributors.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P30 — torch.compile and TorchDynamo**.

1. [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) — learn the API or system contract.
2. [TorchDynamo deep dive](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html) — study the underlying design, paper, or specification.
3. [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

At a high level:

```text
Python frame → Dynamo bytecode analysis → FX graph region + guards
            → backend(gm, example_inputs) → callable
            → cached result if future guards hold
```

Dynamo uses CPython’s frame-evaluation mechanism to intercept execution rather than requiring you to rewrite a model as a static graph. Unsupported Python, data-dependent control flow, opaque extensions, mutation/alias uncertainty, or explicitly disabled code can form graph breaks. A graph break is not automatically wrong: several correctly compiled regions may outperform a risky forced whole graph. `fullgraph=True` is useful as a diagnostic/contract—it asks capture to fail instead of silently segmenting—not as a universal speed flag.

The public entry point is `torch.compile(fn_or_module, backend="inductor", dynamic=..., mode=..., options=...)`. Modes configure tradeoffs such as compilation/autotuning, CUDA graphs, and overhead; inspect available options in the installed build rather than assuming a blog post’s defaults. The first invocation may include tracing, code generation, compilation, autotuning, and CUDA graph capture. Warm latency needs separately timed calls after synchronization. Keep an eager oracle and reset process-local compiler state only when deliberately testing cold behavior; deleting caches indiscriminately destroys evidence.

Compiler observability matters. Use `TORCH_LOGS` categories documented by the installed release (such as graph breaks, guards, recompiles, dynamic shapes, or output code), `torch._dynamo.explain` where supported, and trace artifacts/options to gather evidence. These diagnostics may be internal/unstable, so put them in a pinned lab script, not application logic. Prefer a registered custom operator with fake/autograd support over `torch.compiler.allow_in_graph`; the latter deliberately bypasses Dynamo frontend checks and can create hard-to-debug incorrectness.

## Implementation ladder in `torch-lab/`

1. **Reference:** make `torch-lab/compiler/compile_baseline.py` run an eager decoder/vision block with deterministic inputs and record first call, N warm calls, output/gradient parity, peak memory, and exceptions.
2. **Optimized:** add `compile_variants.py` sweeping `fullgraph`, `dynamic`, and one installed mode/config at a time. Capture graph-break/recompile diagnostics into versioned artifacts; fix one legitimate break through ordinary PyTorch refactoring or a correctly registered op.
3. **Integrated:** compile a bounded serving subgraph (not an entire scheduler) and compare prefill/decode shapes, batch sizes, cache lifetime, and CUDA-graph interaction. Keep an eager escape hatch and a startup/warmup policy.

## Three focused sessions

1. **Lifecycle:** time cold capture, compilation, and warm execution separately. Inspect the returned wrapper and show why one timing number is misleading.
2. **Breaks:** build intentional breaks—tensor `.item()`, Python data-dependent branch, unsupported custom function—and use diagnostics to explain each. Repair only one and prove behavior is unchanged.
3. **Production contract:** make a small compile policy with shape buckets, versioned cache path, eager fallback, logs, and regression test.

## Experiments and deliverables

Add `compile_manifest.json`, diagnostic logs/artifact directory, `test_compile_parity.py`, and `graph_break_notebook.md` (Markdown is preferable to an opaque notebook). Record Python/PyTorch/Triton/CUDA/driver versions, backend/mode/options, input shapes/strides/dtypes, guard/recompile counts, cold/warm timing, generated-kernel count if available, memory, and quality results.

## Common failure modes

- Reporting post-warmup speed without startup cost for a short-lived workload.
- Turning on every mode/option together and losing causality.
- Treating graph breaks as exceptions to suppress rather than contracts to understand.
- Hiding a Python side effect inside `allow_in_graph`.
- Comparing compiled and eager paths with different autocast, sampling, or memory-cache states.

## Mastery gate

Given an eager function with two graph breaks and variable shapes, identify each break from artifacts, decide whether it merits repair, make one safe repair, and present a cold/warm/quality comparison. Explain why `fullgraph=True` succeeds or fails and why that result does not alone establish performance.

## Primary sources, code paths, and OSS bridge

- [`torch.compile` API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [`torch.compiler` user guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html)
- [Dynamo overview](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)
- [`allow_in_graph` warning and guidance](https://docs.pytorch.org/docs/stable/generated/torch.compiler.allow_in_graph.html)
- [PyTorch 2 compiler paper](https://pytorch.org/assets/pytorch2-2.pdf)

Read `torch/_dynamo/`, especially conversion, guards, and backend interfaces, plus their tests. Contributions with a minimized script, expected behavior, logs, and eager parity are highly valued. Start with docs/tests or a well-isolated graph-break issue; do not submit a blanket skip/disable change that merely hides a compiler defect.

## Next

Module 31 makes guard construction, cache reuse, and dynamic-shape choices explicit.
