# Week 01: Reproducible lab and a useful serving baseline

Days 001–007 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Environment manifest, one request record, C++ executable.

**Prerequisite:** Complete the diagnostic in [START_HERE.md](../START_HERE.md).

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-001"></a>

## Day 001 — Map the system

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P01 — Module 01 — Systems Workflow for PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/01-systems-workflow-for-pytorch.md); [P02 — Module 02 — Reproducible Python, CUDA, and Packaging](../originals/pytorch-systems-and-deployment-curriculum/modules/02-reproducible-python-cuda-and-packaging.md); [I01 — Module 01 — The Inference System Map](../originals/inference-engineering-curriculum/modules/01-inference-system-map.md). Focus on map the system, using the task below to select the relevant subsection.

**External entry:** [PyTorch foundations](https://docs.pytorch.org/tutorials/beginner/basics/intro.html). See [resource verification notes](../RESOURCES.md).

**Do:** Draw your current request path and record OS, Python, compiler, RAM, GPU and available phone; create the lab folders described in PROJECTS.md.

**Done when:** Explain model, framework, runtime and hardware boundaries without notes.

**Career block:** List target roles and existing evidence; choose two role families. Use the [career reference](../originals/ai-career-acceleration/weeks/week-01.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-001.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-002"></a>

## Day 002 — First native program

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E01 — Week 01: First C++ programs and an Android baseline](../originals/edge-inference-roadmap/weeks/week-01.md); [P03 — Module 03 — Python Performance and Data Model](../originals/pytorch-systems-and-deployment-curriculum/modules/03-python-performance-and-data-model.md). Focus on first native program, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Compile a C++ executable with functions and typed inputs; calculate model-weight bytes from parameter count and precision.

**Done when:** Check a hand-calculated example and reject a negative input.

**Career block:** Draft one experience bullet supported by a concrete example. Use the [career reference](../originals/ai-career-acceleration/weeks/week-01.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-002.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-003"></a>

## Day 003 — Measurement contract

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on measurement contract, using the task below to select the relevant subsection.

**External entry:** [PyTorch testing overview](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#testing). See [resource verification notes](../RESOURCES.md).

**Do:** Define request start, first output, completion and error fields; write a Python runner against one existing runtime or a small recorded fixture.

**Done when:** Keep one timeout or malformed response as an explicit failed record.

**Career block:** Draft a second resume variant for platform/backend AI roles. Use the [career reference](../originals/ai-career-acceleration/weeks/week-01.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-003.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-004"></a>

## Day 004 — Pin the baseline

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P06 — Module 06 — Framework Landscape and Interoperability](../originals/pytorch-systems-and-deployment-curriculum/modules/06-framework-landscape-and-interoperability.md); [I02 — Module 02 — Reproducible Stack and Toolchain](../originals/inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md). Focus on pin the baseline, using the task below to select the relevant subsection.

**External entry:** [torch.export](https://docs.pytorch.org/docs/stable/export.html). See [resource verification notes](../RESOURCES.md).

**Do:** Choose one already workable runtime and a small public model; record revisions and run three fixed prompts or document the precise installation blocker.

**Done when:** Another shell can find the exact command, assets and versions.

**Career block:** Update the profile headline/about draft using verified experience. Use the [career reference](../originals/ai-career-acceleration/weeks/week-01.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-004.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-005"></a>

## Day 005 — Repeatability

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E01 — Week 01: First C++ programs and an Android baseline](../originals/edge-inference-roadmap/weeks/week-01.md); [P15 — Module 15 — Randomness, Determinism, and Reproducibility](../originals/pytorch-systems-and-deployment-curriculum/modules/15-randomness-determinism-and-reproducibility.md). Focus on repeatability, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Add configuration and seed fields to the runner; compare fresh-process and already-loaded request paths.

**Done when:** State which caches were controlled and which were not.

**Career block:** Choose the baseline report as the first portfolio artifact. Use the [career reference](../originals/ai-career-acceleration/weeks/week-01.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-005.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-006"></a>

## Day 006 — Baseline report

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on baseline report, using the task below to select the relevant subsection.

**External entry:** [PyTorch testing overview](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#testing). See [resource verification notes](../RESOURCES.md).

**Do:** Save raw records and a one-page report; make a minimal README explaining one successful run and one failure.

**Done when:** A reader can regenerate the summary from saved raw data; no speed claim without measurements.

**Career block:** Screen two current roles for geography and fit; manually prepare or send one application if appropriate. Use the [career reference](../originals/ai-career-acceleration/weeks/week-01.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-006.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Reproduce one request or an explicit unavailable-runtime record from a fresh shell. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-007"></a>

## Day 007 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Reproduce one request or an explicit unavailable-runtime record from a fresh shell.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
