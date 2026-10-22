# Project 06 — Upstream-Ready Contribution

**Window:** Continuous, culminating Weeks 39–42 · **Prerequisites:** none;
Modules 59–62 formalize it

Make one contribution-sized change to PyTorch, TorchAO, torchtune, ExecuTorch,
Torch-TensorRT, vLLM, SGLang, llama.cpp, or a tightly related project. The
target is a reviewable patch—not a speculative rewrite.

## Good contribution shapes

- a minimal reproducer and regression test for a real bug;
- documentation that clarifies a versioned/deployment edge case, with a runnable
  example;
- a focused benchmark or test stabilization improvement;
- an operator/export/compile support gap with a narrow implementation and
  comprehensive contract tests;
- a model/configuration port with correctness and performance evidence.

## Required dossier

1. Issue/problem statement and owner/scope research.
2. Smallest reproducer, expected/actual behavior, and environment information.
3. Source-path map and tests inspected before editing.
4. Proposed change, alternatives rejected, and backward-compatibility risk.
5. Tests at the project’s requested granularity; benchmark if performance is
   claimed.
6. Clear commit series, PR description draft, documentation/release-note impact,
   and a response plan for review feedback.

## Mastery gate

An experienced maintainer should be able to clone the repro, understand the
contract, see why the test fails before the patch, and evaluate the change
without relying on private context. If no PR is ready, an issue-quality dossier
is still valuable; do not force an oversized patch merely to claim a merge.

See [the contribution path](../CONTRIBUTING_TO_PYTORCH.md) and the inference
course's [open-source playbook](../../inference-engineering-curriculum/OPEN_SOURCE.md).
