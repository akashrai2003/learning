# Module 60 — PyTorch Testing, CI, and Triage

**Phase:** 9 — PyTorch source and contribution engineering  
**Time:** 15 focused hours  
**Prerequisites:** Modules 05, 18, 42, 59, and a source-map/build manifest.

## Existing inference curriculum bridge

This operationalizes [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [42 — Compiler stack: Dynamo, Inductor, and Triton](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), [43 — Build/JIT/Ninja/CMake/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md), and [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). A benchmark or crash becomes useful upstream only after it is reduced, classified, and tested at the correct boundary.

## Why this matters and outcomes

PyTorch's test surface spans eager semantics, autograd, operator metadata, CPU/CUDA backends, distributed ranks, compiler graphs, export, C++ APIs, documentation, linting, and a large CI matrix. Good triage is a technical skill: establish expected behavior, reduce the reproducer, control version/hardware variables, locate ownership, write the smallest discriminating test, and communicate what is known versus inferred.

You will be able to select a focused local test, create a multi-layer reproducer ladder, distinguish environment failure from framework defect, use logging/minification tools responsibly, interpret CI outcome categories, and prepare a high-signal issue or test patch.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P60 — PyTorch testing, CI, and triage**.

1. [PyTorch contributing guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) — learn the API or system contract.
2. [PyTorch test tree](https://github.com/pytorch/pytorch/tree/main/test) — study the underlying design, paper, or specification.
3. [PyTorch issue tracker](https://github.com/pytorch/pytorch/issues) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch developer notes](https://docs.pytorch.org/docs/stable/community/contribution_guide.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch source repository](https://github.com/pytorch/pytorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch developer discussion](https://dev-discuss.pytorch.org/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Think in an evidence ladder: user workload → minimal script → unit test → backend/compiler/export/distributed specialization → CI job. Each rung should retain the failure/property while removing unrelated dependencies. A compiler issue needs eager correctness plus compiled divergence/graph evidence; an operator issue needs shape/dtype/device/layout/autograd coverage as appropriate; a distributed issue needs rank logs and a timeout-bounded launch. A source build/import problem needs the full loader/build/toolchain manifest before an API-level issue is credible.

Read the current `CONTRIBUTING.md` before copying commands: it documents focused Python test runners, C++ gtest binaries, lint tools, and CI practices, and those conventions evolve. Local success is not a CI matrix proof. Conversely, a red remote job may be an infrastructure or unrelated-flake failure, so inspect logs and rerun policy before changing code. Never hide a test by adding broad skip/xfail logic unless the maintainer-approved contract truly requires it.

For compile/export diagnostics, capture graph break/recompile/export reports and use a minimizer only after preserving the original reproducer and environment. For performance, keep correctness first and report raw timing/profiler evidence. Your SM120 FlashInfer incident is a model triage case: generated Ninja commands and OOM evidence distinguish host-resource pressure from model/runtime correctness. The same discipline prevents misfiled “CUDA issue” reports.

## Implementation ladder in `torch-lab/`

1. **Reference:** make `torch-lab/triage/reproducer_ladder.md` for one known behavior from this course. Include full workload, reduced standalone script, expected/actual result, version manifest, and a no-GPU or CPU control where meaningful.
2. **Optimized:** add a candidate unittest/property test in a local scratch branch or external reproduction repository. It must fail before the hypothetical fix and pass after it; cover the relevant shape/dtype/device/mode matrix narrowly.
3. **Integrated:** write `triage_dossier.md` and `ci_mapping.md`: suspected subsystem/owners, source/test locations, local commands/logs, targeted CI classes, failure classification, non-goals, and a proposed issue title/body. Do not open an issue or PR merely as a course exercise.

## Three focused sessions

1. **Reduction:** shrink a model/runtime failure without deleting the causative contract. Verify the reduced script independently in a clean environment and capture exact versions.
2. **Test placement:** find neighboring tests in PyTorch, learn their setup/skip conventions, and write one focused test that distinguishes the behavior from a nearby false explanation.
3. **CI literacy:** map changed files to likely lint/unit/CUDA/distributed/compiler jobs. Read one public CI failure log or local equivalent and classify it as test assertion, build/ABI, timeout/hang, environmental, or suspected flaky infrastructure.

## Experiments and deliverables

Commit `reproducer_ladder.md`, `repro.py`, `test_candidate.py` or pseudopatch, `environment_manifest.json`, `ci_mapping.md`, and `triage_dossier.md`. Include expected/actual values, exact commands/exit codes, timeouts, seeds, model/input minimization steps, clean-environment result, relevant logs/traces, source ownership links, test matrix, performance scope, and any data/checkpoint redaction necessary for a public report.

## Common failure modes

- Filing a large application log without a runnable reduced script.
- Calling a local environment mismatch a PyTorch regression.
- Reducing away the dtype/layout/shape/concurrency condition that triggers the bug.
- Adding a wide skip/xfail to make CI green.
- Sending generated explanation or code without personally validating it and knowing the test's contract.

## Mastery gate

Create a minimal, timeout-bounded reproducer and a focused regression test for a real or deliberately injected behavior. Show what each control rules out, identify likely source/test ownership, and write an issue-ready evidence package that a maintainer could run without your machine.

## Primary sources, code paths, and OSS bridge

- [PyTorch contribution technical guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
- [PyTorch testing documentation](https://github.com/pytorch/pytorch/tree/main/test)
- [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- [PyTorch issue tracker](https://github.com/pytorch/pytorch/issues)

Read `test/`, `test/cpp/`, `tools/testing/`, CI workflow/configuration files, and nearby tests for the exact subsystem. Start by responding to a maintainer-labeled actionable issue or improving an existing regression test; broad speculative fixes create review work rather than value.

## Next

Module 61 turns good triage into a maintainable contribution workflow and collaboration practice.
