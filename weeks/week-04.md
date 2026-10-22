# Week 04: Autograd, RAII and the first case study

Days 022–028 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Gradient checks, an owning native buffer and baseline report v1.

**Prerequisite:** Pass the relevant gates from [week 03](week-03.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-022"></a>

## Day 022 — Autograd mechanics

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P10 — Module 10 — Autograd Engine and Gradient Modes](../originals/pytorch-systems-and-deployment-curriculum/modules/10-autograd-engine-and-gradient-modes.md). Focus on autograd mechanics, using the task below to select the relevant subsection.

**External entry:** [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html). See [resource verification notes](../RESOURCES.md).

**Do:** Build a tiny differentiable scalar expression and inspect its graph; compare automatic gradients with finite differences.

**Done when:** Check gradients away from nondifferentiable points with a stated tolerance.

**Career block:** Solve a binary-search boundary problem and explain the invariant. Use the [career reference](../originals/ai-career-acceleration/weeks/week-04.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-022.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-023"></a>

## Day 023 — Module state

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P11 — Module 11 — Modules, Parameters, Buffers, Hooks, and State](../originals/pytorch-systems-and-deployment-curriculum/modules/11-modules-parameters-buffers-hooks-and-state.md); [P12 — Module 12 — torch.nn.functional and Dispatch Semantics](../originals/pytorch-systems-and-deployment-curriculum/modules/12-torch-nn-functional-and-dispatch-semantics.md). Focus on module state, using the task below to select the relevant subsection.

**External entry:** [torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html). See [resource verification notes](../RESOURCES.md).

**Do:** Create a tiny nn.Module with a parameter and a buffer; compare train/eval and gradient modes.

**Done when:** Show that eval mode and disabling gradients solve different problems.

**Career block:** Prepare one application using the first case study. Use the [career reference](../originals/ai-career-acceleration/weeks/week-04.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-023.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-024"></a>

## Day 024 — RAII and moves

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E04 — Week 04: Classes, RAII, moves and a minimal tensor](../originals/edge-inference-roadmap/weeks/week-04.md). Focus on raii and moves, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Wrap a vector-backed tensor buffer in a class; define clear copy and move behavior without raw owning pointers.

**Done when:** Copy, move and destruction preserve values and leave no double ownership.

**Career block:** Draft the first benchmark report's public summary. Use the [career reference](../originals/ai-career-acceleration/weeks/week-04.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-024.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-025"></a>

## Day 025 — Mutation and aliases

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P10 — Module 10 — Autograd Engine and Gradient Modes](../originals/pytorch-systems-and-deployment-curriculum/modules/10-autograd-engine-and-gradient-modes.md); [P07 — Module 07 — Tensor Storage, Views, Strides, and Layouts](../originals/pytorch-systems-and-deployment-curriculum/modules/07-tensor-storage-views-strides-and-layouts.md). Focus on mutation and aliases, using the task below to select the relevant subsection.

**External entry:** [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html). See [resource verification notes](../RESOURCES.md).

**Do:** Trigger an isolated in-place autograd failure; repair it by making the intended state transition explicit.

**Done when:** The repair preserves both forward values and the expected gradient.

**Career block:** Check the chosen contribution reproduction against current upstream. Use the [career reference](../originals/ai-career-acceleration/weeks/week-04.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-025.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-026"></a>

## Day 026 — First portfolio case study

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md); [P18 — Module 18 — PyTorch Profiler, Memory, and Diagnostics](../originals/pytorch-systems-and-deployment-curriculum/modules/18-pytorch-profiler-memory-and-diagnostics.md). Focus on first portfolio case study, using the task below to select the relevant subsection.

**External entry:** [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/). See [resource verification notes](../RESOURCES.md).

**Do:** Turn the week-three sweep into a concise context/concurrency report with raw data and failure counts.

**Done when:** Someone can verify every plotted number from the recorded runs.

**Career block:** Rehearse inference-service system design under a memory limit. Use the [career reference](../originals/ai-career-acceleration/weeks/week-04.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-026.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-027"></a>

## Day 027 — Foundation gate

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [E04 — Week 04: Classes, RAII, moves and a minimal tensor](../originals/edge-inference-roadmap/weeks/week-04.md); [P10 — Module 10 — Autograd Engine and Gradient Modes](../originals/pytorch-systems-and-deployment-curriculum/modules/10-autograd-engine-and-gradient-modes.md). Focus on foundation gate, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Rebuild one ownership example and one gradient example from an empty file; fix the weakest report section.

**Done when:** Pass the week gate before adding checkpoint or native integration work.

**Career block:** Audit the first month's interview/application evidence and change one weak part of the funnel. Use the [career reference](../originals/ai-career-acceleration/weeks/week-04.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-027.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain backward state and resource cleanup; reproduce the first report. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-028"></a>

## Day 028 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain backward state and resource cleanup; reproduce the first report.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
