# Week 16: Guards, functionalization and compiler debugging

Days 106–112 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Dynamic-shape matrix and minimized graph-break case.

**Prerequisite:** Pass the relevant gates from [week 15](week-15.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-106"></a>

## Day 106 — Guards

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P31 — Module 31 — Guards, Recompilation, and Dynamic Shapes](../originals/pytorch-systems-and-deployment-curriculum/modules/31-guards-recompilation-and-dynamic-shapes.md). Focus on guards, using the task below to select the relevant subsection.

**External entry:** [Dynamic shapes manual](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html). See [resource verification notes](../RESOURCES.md).

**Do:** Vary shape, dtype and one Python value separately; inspect recompilation behavior.

**Done when:** Map each observed recompile to an input change or document the unknown.

**Career block:** Audit actual pipeline outcomes and select the highest-impact next action. Use the [career reference](../originals/ai-career-acceleration/weeks/week-16.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-106.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-107"></a>

## Day 107 — Dynamic shapes

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P31 — Module 31 — Guards, Recompilation, and Dynamic Shapes](../originals/pytorch-systems-and-deployment-curriculum/modules/31-guards-recompilation-and-dynamic-shapes.md); [P51 — Module 51 — `torch.export` and Program Capture](../originals/pytorch-systems-and-deployment-curriculum/modules/51-torch-export-and-program-capture.md). Focus on dynamic shapes, using the task below to select the relevant subsection.

**External entry:** [Dynamic shapes manual](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html). See [resource verification notes](../RESOURCES.md).

**Do:** Declare a small supported shape contract and compare fixed versus varying lengths.

**Done when:** Test an allowed boundary and an explicitly rejected input.

**Career block:** Reconstruct a technical interview answer without notes. Use the [career reference](../originals/ai-career-acceleration/weeks/week-16.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-107.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-108"></a>

## Day 108 — Mutation and functionalization

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P32 — Module 32 — AOTAutograd, Functionalization, and Autograd Graphs](../originals/pytorch-systems-and-deployment-curriculum/modules/32-aotautograd-functionalization-and-autograd-graphs.md); [P07 — Module 07 — Tensor Storage, Views, Strides, and Layouts](../originals/pytorch-systems-and-deployment-curriculum/modules/07-tensor-storage-views-strides-and-layouts.md). Focus on mutation and functionalization, using the task below to select the relevant subsection.

**External entry:** [AOTAutograd source](https://github.com/pytorch/pytorch/tree/main/torch/_functorch). See [resource verification notes](../RESOURCES.md).

**Do:** Compare an in-place toy computation with a functional equivalent.

**Done when:** Values and intended alias behavior remain explicit across capture.

**Career block:** Update capability statements to match demonstrated work. Use the [career reference](../originals/ai-career-acceleration/weeks/week-16.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-108.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-109"></a>

## Day 109 — Backward capture

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P32 — Module 32 — AOTAutograd, Functionalization, and Autograd Graphs](../originals/pytorch-systems-and-deployment-curriculum/modules/32-aotautograd-functionalization-and-autograd-graphs.md); [P10 — Module 10 — Autograd Engine and Gradient Modes](../originals/pytorch-systems-and-deployment-curriculum/modules/10-autograd-engine-and-gradient-modes.md). Focus on backward capture, using the task below to select the relevant subsection.

**External entry:** [AOTAutograd source](https://github.com/pytorch/pytorch/tree/main/torch/_functorch). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect a tiny training graph through AOTAutograd; relate saved state to the autograd baseline.

**Done when:** Explain the forward/backward boundary using the actual example.

**Career block:** Compare any real offers using the existing offer guide; otherwise improve targeting. Use the [career reference](../originals/ai-career-acceleration/weeks/week-16.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-109.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-110"></a>

## Day 110 — Minimization

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P42 — Module 42 — Compiler Debugging, Minimization, and Regression Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/42-compiler-debugging-minimization-and-regression.md). Focus on minimization, using the task below to select the relevant subsection.

**External entry:** [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html). See [resource verification notes](../RESOURCES.md).

**Do:** Reduce one graph break or observed compiler error to a short script.

**Done when:** The minimal script reproduces the behavior in a fresh process.

**Career block:** Write a four-week career extension with specific actions. Use the [career reference](../originals/ai-career-acceleration/weeks/week-16.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-110.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-111"></a>

## Day 111 — Compiler gate

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P31 — Module 31 — Guards, Recompilation, and Dynamic Shapes](../originals/pytorch-systems-and-deployment-curriculum/modules/31-guards-recompilation-and-dynamic-shapes.md); [P42 — Module 42 — Compiler Debugging, Minimization, and Regression Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/42-compiler-debugging-minimization-and-regression.md). Focus on compiler gate, using the task below to select the relevant subsection.

**External entry:** [Dynamic shapes manual](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html). See [resource verification notes](../RESOURCES.md).

**Do:** Package the guard matrix, graph-break explanation and fallback into the report.

**Done when:** Explain whether compilation helps the real shape distribution rather than one convenient input.

**Career block:** Review the 16-week career phase; if interviews are active, choose the career-priority pacing option. Use the [career reference](../originals/ai-career-acceleration/weeks/week-16.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-111.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain one guard and minimize a compiler failure. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-112"></a>

## Day 112 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain one guard and minimize a compiler failure.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
