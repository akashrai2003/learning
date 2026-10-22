# Week 03: Numerical reasoning, pointers and ownership

Days 015–021 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Numerical-error notebook and a native debugging note.

**Prerequisite:** Pass the relevant gates from [week 02](week-02.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-015"></a>

## Day 015 — Floating-point error

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P04 — Module 04 — Tensor Math and Numerical Contracts](../originals/pytorch-systems-and-deployment-curriculum/modules/04-tensor-math-and-numerical-contracts.md); [I04 — Module 04 — Numerical Formats and Error](../originals/inference-engineering-curriculum/modules/04-numerical-formats-and-error.md). Focus on floating-point error, using the task below to select the relevant subsection.

**External entry:** [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html). See [resource verification notes](../RESOURCES.md).

**Do:** Compare FP32 and available lower-precision reductions with a float64 oracle; vary magnitude and summation order.

**Done when:** Record absolute and relative error with a denominator policy near zero.

**Career block:** Implement a small top-k/heap problem and test boundary cases. Use the [career reference](../originals/ai-career-acceleration/weeks/week-03.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-015.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-016"></a>

## Day 016 — References and pointers

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E03 — Week 03: References, pointers, containers and debugging](../originals/edge-inference-roadmap/weeks/week-03.md). Focus on references and pointers, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Implement read-only and mutating vector functions; draw ownership and lifetime for each argument.

**Done when:** Explain every reference and pointer lifetime in the sample.

**Career block:** Screen and act on one high-fit role; record the next action. Use the [career reference](../originals/ai-career-acceleration/weeks/week-03.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-016.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-017"></a>

## Day 017 — Debugger and sanitizer

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E03 — Week 03: References, pointers, containers and debugging](../originals/edge-inference-roadmap/weeks/week-03.md); [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md). Focus on debugger and sanitizer, using the task below to select the relevant subsection.

**External entry:** [Clang AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html). See [resource verification notes](../RESOURCES.md).

**Do:** Create one small bounds/lifetime bug in an isolated program; locate and repair it with a debugger or sanitizer.

**Done when:** Preserve the diagnostic, root cause and passing regression case.

**Career block:** Turn a parser failure into a brief technical explanation. Use the [career reference](../originals/ai-career-acceleration/weeks/week-03.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-017.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-018"></a>

## Day 018 — Devices and memory

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P09 — Module 09 — Devices, Dtypes, Memory, and Streams](../originals/pytorch-systems-and-deployment-curriculum/modules/09-devices-dtypes-memory-and-streams.md); [I05 — Module 05 — Performance Modeling and Roofline](../originals/inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md). Focus on devices and memory, using the task below to select the relevant subsection.

**External entry:** [CUDA semantics in PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html). See [resource verification notes](../RESOURCES.md).

**Do:** Measure a CPU operation and available GPU operation separately from transfers; estimate arithmetic intensity.

**Done when:** Do not attribute transfer or launch time to arithmetic speed.

**Career block:** Shortlist one reproducible test/docs contribution opportunity. Use the [career reference](../originals/ai-career-acceleration/weeks/week-03.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-018.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-019"></a>

## Day 019 — Controlled workload sweep

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P18 — Module 18 — PyTorch Profiler, Memory, and Diagnostics](../originals/pytorch-systems-and-deployment-curriculum/modules/18-pytorch-profiler-memory-and-diagnostics.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on controlled workload sweep, using the task below to select the relevant subsection.

**External entry:** [Profiler API](https://docs.pytorch.org/docs/stable/profiler.html). See [resource verification notes](../RESOURCES.md).

**Do:** Run two prompt lengths and two concurrency settings using the existing runner; retain failures and memory definitions.

**Done when:** Compare only runs with the same model, output policy and recorded setup.

**Career block:** Explain async cancellation and backpressure using your runner. Use the [career reference](../originals/ai-career-acceleration/weeks/week-03.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-019.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-020"></a>

## Day 020 — Reconstruction

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [E03 — Week 03: References, pointers, containers and debugging](../originals/edge-inference-roadmap/weeks/week-03.md); [P04 — Module 04 — Tensor Math and Numerical Contracts](../originals/pytorch-systems-and-deployment-curriculum/modules/04-tensor-math-and-numerical-contracts.md). Focus on reconstruction, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Rewrite the native dot product and error metric without looking; annotate the report with numeric limitations.

**Done when:** Explain why lower bit width need not mean faster execution.

**Career block:** Complete a coding mock with review and update the strongest current application. Use the [career reference](../originals/ai-career-acceleration/weeks/week-03.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-020.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain precision versus range and repair one lifetime bug. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-021"></a>

## Day 021 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain precision versus range and repair one lifetime bug.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
