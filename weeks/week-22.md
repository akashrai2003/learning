# Week 22: CUDA Graphs, profiling and optimization integration

Days 148–154 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** One end-to-end optimized path and regression report.

**Prerequisite:** Pass the relevant gates from [week 21](week-21.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-148"></a>

## Day 148 — Graph capture constraints

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P41 — Module 41 — CUDA Graphs, Memory, and Framework Integration](../originals/pytorch-systems-and-deployment-curriculum/modules/41-cuda-graphs-memory-and-framework-integration.md). Focus on graph capture constraints, using the task below to select the relevant subsection.

**External entry:** [PyTorch CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#constraints). See [resource verification notes](../RESOURCES.md).

**Do:** Capture one small supported inference path with fixed buffers; retain an eager fallback.

**Done when:** Inputs/outputs remain correct across repeated replay and state reset.

**Career block:** Recall one coding or debugging pattern relevant to end-to-end profiling; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-148.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-149"></a>

## Day 149 — Timeline first

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P18 — Module 18 — PyTorch Profiler, Memory, and Diagnostics](../originals/pytorch-systems-and-deployment-curriculum/modules/18-pytorch-profiler-memory-and-diagnostics.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on timeline first, using the task below to select the relevant subsection.

**External entry:** [Profiler API](https://docs.pytorch.org/docs/stable/profiler.html). See [resource verification notes](../RESOURCES.md).

**Do:** Collect a bounded profiler trace of the chosen decoder or serving workload.

**Done when:** Identify time spent in copies, allocations, launches and computation.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to end-to-end profiling. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-149.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-150"></a>

## Day 150 — One hypothesis

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md); [I05 — Module 05 — Performance Modeling and Roofline](../originals/inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md). Focus on one hypothesis, using the task below to select the relevant subsection.

**External entry:** [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/). See [resource verification notes](../RESOURCES.md).

**Do:** Select the highest-impact measured bottleneck and write a falsifiable optimization hypothesis.

**Done when:** The experiment changes only the relevant variable.

**Career block:** Add one verified end-to-end profiling observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-150.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-151"></a>

## Day 151 — Integration

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P41 — Module 41 — CUDA Graphs, Memory, and Framework Integration](../originals/pytorch-systems-and-deployment-curriculum/modules/41-cuda-graphs-memory-and-framework-integration.md); [I28 — Module 28 — Quantized Kernel and System Integration](../originals/inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md). Focus on integration, using the task below to select the relevant subsection.

**External entry:** [PyTorch CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#constraints). See [resource verification notes](../RESOURCES.md).

**Do:** Integrate one verified kernel or quantized operation into the full path.

**Done when:** Full-path outputs and quality pass the earlier acceptance thresholds.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-151.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-152"></a>

## Day 152 — Regression and startup

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P42 — Module 42 — Compiler Debugging, Minimization, and Regression Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/42-compiler-debugging-minimization-and-regression.md); [I43 — Module 43 — Build, JIT, Ninja, CMake, and ABI](../originals/inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md). Focus on regression and startup, using the task below to select the relevant subsection.

**External entry:** [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html). See [resource verification notes](../RESOURCES.md).

**Do:** Compare clean startup, compile/build time and warm execution; reproduce in an isolated output directory.

**Done when:** Report host-memory and device-memory effects separately where measurable.

**Career block:** Explain end-to-end profiling aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-152.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-153"></a>

## Day 153 — GPU/compiler gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P18 — Module 18 — PyTorch Profiler, Memory, and Diagnostics](../originals/pytorch-systems-and-deployment-curriculum/modules/18-pytorch-profiler-memory-and-diagnostics.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on gpu/compiler gate, using the task below to select the relevant subsection.

**External entry:** [Profiler API](https://docs.pytorch.org/docs/stable/profiler.html). See [resource verification notes](../RESOURCES.md).

**Do:** Write one optimization report including a failure or non-improvement case.

**Done when:** The result is reproducible and any remaining hardware limitation is explicit.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use end-to-end profiling as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-153.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Show where the bottleneck moved or why optimization did not help. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-154"></a>

## Day 154 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Show where the bottleneck moved or why optimization did not help.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
