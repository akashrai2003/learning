# Week 02: Tensor shapes, layouts and C++ control flow

Days 008–014 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Shape notebook, layout inspector and runner validation.

**Prerequisite:** Pass the relevant gates from [week 01](week-01.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-008"></a>

## Day 008 — Address arithmetic

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P07 — Module 07 — Tensor Storage, Views, Strides, and Layouts](../originals/pytorch-systems-and-deployment-curriculum/modules/07-tensor-storage-views-strides-and-layouts.md); [P08 — Module 08 — Shapes, Indexing, Broadcasting, and einsum](../originals/pytorch-systems-and-deployment-curriculum/modules/08-shapes-indexing-broadcasting-and-einsum.md); [I03 — Module 03 — Tensor Math, Shapes, and Einstein Notation](../originals/inference-engineering-curriculum/modules/03-tensor-math-shapes-and-einsum.md). Focus on address arithmetic, using the task below to select the relevant subsection.

**External entry:** [Tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html). See [resource verification notes](../RESOURCES.md).

**Do:** Build a layout inspector for shape, stride and offset; hand-compute addresses for a small tensor and its transpose.

**Done when:** Predictions match actual indexing for five cases.

**Career block:** Solve one short dictionary/array problem and explain complexity. Use the [career reference](../originals/ai-career-acceleration/weeks/week-02.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-008.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-009"></a>

## Day 009 — Broadcasting and loops

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P08 — Module 08 — Shapes, Indexing, Broadcasting, and einsum](../originals/pytorch-systems-and-deployment-curriculum/modules/08-shapes-indexing-broadcasting-and-einsum.md); [E02 — Week 02: Control flow, numeric types and reproducible experiments](../originals/edge-inference-roadmap/weeks/week-02.md). Focus on broadcasting and loops, using the task below to select the relevant subsection.

**External entry:** [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html). See [resource verification notes](../RESOURCES.md).

**Do:** Implement a C++ dot product with loops and argument checks; implement the same shape contract in PyTorch.

**Done when:** Compare odd lengths, length one and mismatched inputs against a hand oracle.

**Career block:** Qualify two roles and record eligibility evidence from current postings. Use the [career reference](../originals/ai-career-acceleration/weeks/week-02.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-009.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-010"></a>

## Day 010 — Views and copies

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P07 — Module 07 — Tensor Storage, Views, Strides, and Layouts](../originals/pytorch-systems-and-deployment-curriculum/modules/07-tensor-storage-views-strides-and-layouts.md); [P12 — Module 12 — torch.nn.functional and Dispatch Semantics](../originals/pytorch-systems-and-deployment-curriculum/modules/12-torch-nn-functional-and-dispatch-semantics.md). Focus on views and copies, using the task below to select the relevant subsection.

**External entry:** [Tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html). See [resource verification notes](../RESOURCES.md).

**Do:** Compare slice, transpose, reshape, clone and expand; record alias behavior before using these in model code.

**Done when:** Explain one view failure and one hidden allocation.

**Career block:** Draft a short baseline-method note; publish only if useful and ready. Use the [career reference](../originals/ai-career-acceleration/weeks/week-02.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-010.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-011"></a>

## Day 011 — Streaming records

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P03 — Module 03 — Python Performance and Data Model](../originals/pytorch-systems-and-deployment-curriculum/modules/03-python-performance-and-data-model.md); [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md). Focus on streaming records, using the task below to select the relevant subsection.

**External entry:** [Python data model](https://docs.python.org/3/reference/datamodel.html). See [resource verification notes](../RESOURCES.md).

**Do:** Extend the baseline parser with fragmented events and delayed first output using a local fixture.

**Done when:** Malformed events and midstream failure remain distinguishable from success.

**Career block:** Read contribution guidance for one primary repo and record the policy date. Use the [career reference](../originals/ai-career-acceleration/weeks/week-02.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-011.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-012"></a>

## Day 012 — Integer and float arithmetic

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E02 — Week 02: Control flow, numeric types and reproducible experiments](../originals/edge-inference-roadmap/weeks/week-02.md); [P04 — Module 04 — Tensor Math and Numerical Contracts](../originals/pytorch-systems-and-deployment-curriculum/modules/04-tensor-math-and-numerical-contracts.md). Focus on integer and float arithmetic, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Write a size estimator using checked integer arithmetic and compare float32 reductions with float64.

**Done when:** Show a safe overflow rejection and one rounding counterexample.

**Career block:** Rehearse one backend incident using situation, decision and result. Use the [career reference](../originals/ai-career-acceleration/weeks/week-02.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-012.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-013"></a>

## Day 013 — Layout benchmark

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P07 — Module 07 — Tensor Storage, Views, Strides, and Layouts](../originals/pytorch-systems-and-deployment-curriculum/modules/07-tensor-storage-views-strides-and-layouts.md); [I05 — Module 05 — Performance Modeling and Roofline](../originals/inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md). Focus on layout benchmark, using the task below to select the relevant subsection.

**External entry:** [Tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html). See [resource verification notes](../RESOURCES.md).

**Do:** Measure contiguous versus transposed matrix multiplication at a few small shapes; include conversion cost.

**Done when:** The report distinguishes metadata operations from data copies.

**Career block:** Run a 30-minute design mock; spend the remaining block on one targeted application and tracker review. Use the [career reference](../originals/ai-career-acceleration/weeks/week-02.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-013.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Predict five tensor layouts and explain one measured copy. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-014"></a>

## Day 014 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Predict five tensor layouts and explain one measured copy.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
