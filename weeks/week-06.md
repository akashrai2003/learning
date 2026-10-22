# Week 06: Data pipelines, minimal training and Python/native boundaries

Days 036–042 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Tiny training loop and one safely bound native operator.

**Prerequisite:** Pass the relevant gates from [week 05](week-05.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-036"></a>

## Day 036 — Data and batches

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P13 — Module 13 — Data Pipelines, Datasets, and DataLoaders](../originals/pytorch-systems-and-deployment-curriculum/modules/13-data-pipelines-datasets-and-dataloaders.md). Focus on data and batches, using the task below to select the relevant subsection.

**External entry:** [Data loading tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html). See [resource verification notes](../RESOURCES.md).

**Do:** Create a tiny dataset and collate function with explicit shapes; compare zero and available worker counts.

**Done when:** Check missing samples, last batches and deterministic evaluation order.

**Career block:** Practice graph traversal on a small bounded problem. Use the [career reference](../originals/ai-career-acceleration/weeks/week-06.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-036.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-037"></a>

## Day 037 — Minimal training loop

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P14 — Module 14 — Optimizers, Schedulers, and Training State](../originals/pytorch-systems-and-deployment-curriculum/modules/14-optimizers-schedulers-and-training-state.md); [P19 — Module 19 — Minimal Training Engine](../originals/pytorch-systems-and-deployment-curriculum/modules/19-minimal-training-engine.md). Focus on minimal training loop, using the task below to select the relevant subsection.

**External entry:** [torch.optim](https://docs.pytorch.org/docs/stable/optim.html). See [resource verification notes](../RESOURCES.md).

**Do:** Train a small model on a tiny synthetic task with optimizer, evaluation and checkpoint steps.

**Done when:** Loss improves on the toy task and evaluation does not mutate training state.

**Career block:** Qualify one role and complete its next application action. Use the [career reference](../originals/ai-career-acceleration/weeks/week-06.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-037.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-038"></a>

## Day 038 — Mixed precision

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P22 — Module 22 — Automatic Mixed Precision and Numerical Stability](../originals/pytorch-systems-and-deployment-curriculum/modules/22-automatic-mixed-precision-and-numerical-stability.md). Focus on mixed precision, using the task below to select the relevant subsection.

**External entry:** [AMP documentation](https://docs.pytorch.org/docs/stable/amp.html). See [resource verification notes](../RESOURCES.md).

**Do:** Compare FP32 and supported AMP on the same tiny step; inspect loss and gradient behavior.

**Done when:** Record unsupported devices and distinguish stability from speed.

**Career block:** Explain one test that caught a real bug in the lab. Use the [career reference](../originals/ai-career-acceleration/weeks/week-06.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-038.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-039"></a>

## Day 039 — Binding contract

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E06 — Week 06: Concurrency, Python bindings and systems foundations](../originals/edge-inference-roadmap/weeks/week-06.md); [P39 — Module 39 — LibTorch and the C++ Frontend](../originals/pytorch-systems-and-deployment-curriculum/modules/39-libtorch-and-cpp-frontend.md). Focus on binding contract, using the task below to select the relevant subsection.

**External entry:** [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/). See [resource verification notes](../RESOURCES.md).

**Do:** Bind the existing C++ dot product to Python using the edge guide; specify dtype, contiguity and ownership.

**Done when:** Compare against NumPy/PyTorch and reject invalid shapes safely.

**Career block:** Practice one SQL join/aggregation exercise. Use the [career reference](../originals/ai-career-acceleration/weeks/week-06.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-039.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-040"></a>

## Day 040 — Concurrency and GIL

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E06 — Week 06: Concurrency, Python bindings and systems foundations](../originals/edge-inference-roadmap/weeks/week-06.md); [P03 — Module 03 — Python Performance and Data Model](../originals/pytorch-systems-and-deployment-curriculum/modules/03-python-performance-and-data-model.md). Focus on concurrency and gil, using the task below to select the relevant subsection.

**External entry:** [pybind11 advanced functions and GIL](https://pybind11.readthedocs.io/en/stable/advanced/misc.html). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect one native call boundary and implement a bounded worker example; identify synchronization and lifetime needs.

**Done when:** Cancellation or worker completion leaves owned buffers valid.

**Career block:** Design a serving API with timeouts and queue limits. Use the [career reference](../originals/ai-career-acceleration/weeks/week-06.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-040.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-041"></a>

## Day 041 — Training gate and transforms

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P17 — Module 17 — torch.func, vmap, and Functional Transforms](../originals/pytorch-systems-and-deployment-curriculum/modules/17-torch-func-vmap-and-functional-transforms.md); [P19 — Module 19 — Minimal Training Engine](../originals/pytorch-systems-and-deployment-curriculum/modules/19-minimal-training-engine.md). Focus on training gate and transforms, using the task below to select the relevant subsection.

**External entry:** [torch.func](https://docs.pytorch.org/docs/stable/func.html). See [resource verification notes](../RESOURCES.md).

**Do:** Use one vmap or grad transform on the tiny model; produce a training-loop and native-boundary walkthrough.

**Done when:** Explain parameters, gradients, optimizer state and sample batching without notes.

**Career block:** Run a timed mock and review the six-week funnel; narrow roles if eligibility is repeatedly wrong. Use the [career reference](../originals/ai-career-acceleration/weeks/week-06.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-041.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Resume training and explain buffer ownership across the binding. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-042"></a>

## Day 042 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Resume training and explain buffer ownership across the binding.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
