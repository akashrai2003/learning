# Week 15: FX graphs and torch.compile

Days 099–105 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Eager/compiled comparison with an annotated graph.

**Prerequisite:** Pass the relevant gates from [week 14](week-14.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-099"></a>

## Day 099 — FX graph

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P29 — Module 29 — FX Graphs and Program Transforms](../originals/pytorch-systems-and-deployment-curriculum/modules/29-fx-graphs-and-program-transforms.md). Focus on fx graph, using the task below to select the relevant subsection.

**External entry:** [FX documentation](https://docs.pytorch.org/docs/stable/fx.html). See [resource verification notes](../RESOURCES.md).

**Do:** Trace a small module and annotate nodes, shapes and mutable state.

**Done when:** The traced module matches eager output on fixed inputs.

**Career block:** Repair the weakest coding pattern observed so far. Use the [career reference](../originals/ai-career-acceleration/weeks/week-15.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-099.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-100"></a>

## Day 100 — First compilation

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P30 — Module 30 — `torch.compile` and TorchDynamo](../originals/pytorch-systems-and-deployment-curriculum/modules/30-torch-compile-and-torchdynamo.md); [I42 — Module 42 — Compiler Stack: Dynamo, Inductor, and Triton](../originals/inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md). Focus on first compilation, using the task below to select the relevant subsection.

**External entry:** [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html). See [resource verification notes](../RESOURCES.md).

**Do:** Compile a small decoder block or linear/norm chain with a pinned PyTorch version.

**Done when:** Record cold compile time separately from warmed execution.

**Career block:** Prepare a high-fit application with the canonical case study. Use the [career reference](../originals/ai-career-acceleration/weeks/week-15.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-100.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-101"></a>

## Day 101 — One graph rewrite

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P29 — Module 29 — FX Graphs and Program Transforms](../originals/pytorch-systems-and-deployment-curriculum/modules/29-fx-graphs-and-program-transforms.md). Focus on one graph rewrite, using the task below to select the relevant subsection.

**External entry:** [FX documentation](https://docs.pytorch.org/docs/stable/fx.html). See [resource verification notes](../RESOURCES.md).

**Do:** Apply a small semantics-preserving rewrite to a toy graph, using a clear reference.

**Done when:** Test positive and negative inputs plus more than one shape.

**Career block:** Draft a short case-study summary tied to actual results. Use the [career reference](../originals/ai-career-acceleration/weeks/week-15.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-101.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-102"></a>

## Day 102 — Compiler measurements

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P30 — Module 30 — `torch.compile` and TorchDynamo](../originals/pytorch-systems-and-deployment-curriculum/modules/30-torch-compile-and-torchdynamo.md); [P18 — Module 18 — PyTorch Profiler, Memory, and Diagnostics](../originals/pytorch-systems-and-deployment-curriculum/modules/18-pytorch-profiler-memory-and-diagnostics.md). Focus on compiler measurements, using the task below to select the relevant subsection.

**External entry:** [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html). See [resource verification notes](../RESOURCES.md).

**Do:** Profile eager and compiled paths under the same shape distribution.

**Done when:** Include warm-up, synchronization and memory allocation boundaries.

**Career block:** Rehearse a negotiation conversation using hypothetical terms clearly labeled. Use the [career reference](../originals/ai-career-acceleration/weeks/week-15.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-102.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-103"></a>

## Day 103 — Generated code

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P33 — Module 33 — TorchInductor: Fusion, Scheduling, and Code Generation](../originals/pytorch-systems-and-deployment-curriculum/modules/33-torchinductor-fusion-scheduling-and-codegen.md). Focus on generated code, using the task below to select the relevant subsection.

**External entry:** [TorchInductor docs](https://docs.pytorch.org/docs/stable/torch.compiler.html). See [resource verification notes](../RESOURCES.md).

**Do:** Locate one generated kernel and trace it back to the graph operation.

**Done when:** Identify one fusion and one boundary without claiming full compiler mastery.

**Career block:** Run a focused interview rehearsal with follow-up questions. Use the [career reference](../originals/ai-career-acceleration/weeks/week-15.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-103.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-104"></a>

## Day 104 — Compile report

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P30 — Module 30 — `torch.compile` and TorchDynamo](../originals/pytorch-systems-and-deployment-curriculum/modules/30-torch-compile-and-torchdynamo.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on compile report, using the task below to select the relevant subsection.

**External entry:** [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html). See [resource verification notes](../RESOURCES.md).

**Do:** Write the smallest reproducible eager-versus-compiled experiment with a fallback command.

**Done when:** A clean run reproduces correctness and separates compile overhead from throughput.

**Career block:** Audit resume evidence, demo reliability and active opportunities before final-round work. Use the [career reference](../originals/ai-career-acceleration/weeks/week-15.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-104.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain capture boundaries and separate compile time from steady state. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-105"></a>

## Day 105 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain capture boundaries and separate compile time from steady state.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
