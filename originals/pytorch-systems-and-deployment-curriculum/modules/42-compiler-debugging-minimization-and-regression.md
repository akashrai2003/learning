# Module 42 — Compiler Debugging, Minimization, and Regression Engineering

**Phase:** 6 — Native PyTorch and GPU integration  
**Time:** 15 focused hours  
**Prerequisites:** Modules 29–41, unit testing, profiling, version control, and humility about evidence.

## Existing inference curriculum bridge

This unifies [02 — Reproducible stack/toolchain](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), [28 — Quantized integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), [43 — Build/JIT/Ninja/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md), [44 — Profiling/benchmarking/correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md), and [46 — vLLM contributions](../../inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md).

## Why this matters and outcomes

Compiler work becomes valuable when a difficult failure is reduced to a trustworthy report or regression test. “`torch.compile` is slow/broken” cannot be acted on. A 40-line script with pinned environment, eager oracle, exact input contract, artifact bundle, and one clear expected-versus-actual result can become an upstream fix.

You will learn a layered diagnosis method, use compiler observability without mistaking private diagnostics for app APIs, minimize semantic and performance regressions, bisect responsibly, and add tests that prevent recurrence across PyTorch, TorchAO, vLLM, SGLang, or your own extensions.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P42 — Compiler debugging, minimization, and regression**.

1. [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) — learn the API or system contract.
2. [TorchDynamo FAQ](https://docs.pytorch.org/docs/stable/torch.compiler_faq.html) — study the underlying design, paper, or specification.
3. [Dynamo repro tooling source](https://github.com/pytorch/pytorch/tree/main/torch/_dynamo) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch custom-ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch extension-cpp examples](https://github.com/pytorch/extension-cpp) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: classify before changing

First classify the symptom:

| Layer | Evidence question |
| --- | --- |
| eager semantics | Does a compact eager oracle already fail? |
| Dynamo capture | Does `backend="eager"` fail? |
| AOTAutograd | Does an installed `aot_eager` debug backend fail for training? |
| Inductor/lowering | Does the failure begin only with Inductor? |
| native build/loader | Is it compile/link/import/driver-JIT rather than graph semantics? |
| runtime kernel | Is there a CUDA error, wrong result, race, or memory fault? |
| performance | Which stage regressed: cold compile, recompile, launch, kernel, memory, queueing? |

The documented backend ladder isolates components: eager execution is the oracle; Dynamo-only/eager backend tests capture; AOT-eager checks forward/backward capture; Inductor adds lowering/codegen. Exact debug backends/options are release-dependent, so list them from the installed build and write them into your manifest. A custom op should be isolated as reference composition → Python custom op → native/Triton implementation; this catches metadata/alias/stream errors before blaming the compiler.

Correctness minimization preserves the smallest input/model/operation sequence that still disagrees, including dtype, stride, grad mode, seed, shape, device, processor/tokenizer if relevant, and tolerances. Performance minimization preserves the workload distribution and warmup/cache state that triggers the regression; reducing it to one microkernel may eliminate the problem. Use binary search over code, shapes, releases, or flags only when the environment is pinned and a result is repeatable. A bisect that changes CUDA driver, PyTorch wheel, and model revision concurrently is not a bisect.

`TORCH_LOGS`, `TORCH_TRACE`/parsing tools, `TORCH_COMPILE_DEBUG`, trace/source artifacts, profiler timelines, and compiler minifiers are powerful diagnosis aids. Turn them on narrowly, archive paths/version/options, redact secrets, and turn them off in performance measurements. Never paper over an incorrect graph by globally suppressing errors or disabling compilation without recording it as a temporary mitigation.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/debug/repro_template.py` with deterministic input generation, eager oracle, configurable backend/mode, numerical/gradient checks, timing separation, environment manifest, and one command to run it in a clean process.
2. **Optimized:** seed a controlled failure set: graph break/recompile storm, fake-meta mismatch, wrong custom-op alias metadata, CUDA current-stream race, JIT host OOM (or safe simulation), and an Inductor numerical/performance issue. Reduce each to the smallest artifact while retaining the defining symptom.
3. **Integrated:** take one real issue from PyTorch/TorchAO/vLLM/SGLang, reproduce it at a pinned revision, inspect source/logs, propose a test or documentation correction, and draft an issue/PR only after checking project contribution rules and duplicate reports.

## Three focused sessions

1. **Layer isolation:** run one model through eager, capture-only, AOT, and Inductor steps; record exactly where failure appears and why that narrows ownership.
2. **Reduction:** use structured deletion/bisection to shrink a failure while preserving shape/stride/state and a fixed assertion. Compare a correct minimal semantic repro with a performance repro.
3. **Regression craft:** convert one repro into a fast, deterministic test; add a benchmark guard only with variance-aware threshold and explain expected failure mode in a review-ready description.

## Experiments and deliverables

Commit `repro_template.py`, `environment_lock.txt`, `repros/`, `triage_matrix.md`, `minimization_log.md`, `regression_test_plan.md`, and raw benchmark traces. Every report includes exact command, source commit/diff, Python/PyTorch/Triton/CUDA/driver/compiler/GPU data, model/data hashes, input contract, expected/actual, full stack/error, artifact links, and which layers have been eliminated.

## Common failure modes

- Starting with a huge production server log and no independently runnable script.
- Reducing away the dynamic shape, noncontiguous stride, stream, cache, or processor condition that caused failure.
- Calling a fallback-to-eager path a compiler fix.
- Changing tolerance until a numerical failure disappears without error distribution/task analysis.
- Making benchmark CI flaky by setting a hard threshold from one noisy sample.

## Mastery gate

Take a real or seeded compiler/native failure and produce a clean repro, layer-classification table, minimized source, artifact bundle, root-cause hypothesis bounded by evidence, and a regression test plan. A reviewer unfamiliar with your machine should be able to run it, see the failure, and understand what change would validate the fix.

## Primary sources, code paths, and OSS bridge

- [PyTorch `torch.compile` troubleshooting guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
- [PyTorch compiler FAQ/debug ladder](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html)
- [`torch.compile` API and guard diagnostics](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
- [vLLM contributing guide](https://docs.vllm.ai/en/latest/contributing/)

Read `torch/_dynamo/repro/`, `torch/_inductor/`, compiler test suites, `torch/testing/`, and the target project’s issue templates/contribution docs. The highest-leverage OSS contribution is often a minimal regression test, a correct diagnosis, or a benchmark artifact—not a broad rewrite. Preserve the eager reference and make correctness criteria explicit before discussing speed.

## Next

Continue into the deployment/edge modules and the projects track, using this debugging template for every cross-framework or upstream contribution.
