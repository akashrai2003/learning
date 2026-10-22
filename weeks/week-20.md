# Week 20: GEMM, tiling and fusion

Days 134–140 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Small tiled GEMM experiment and library comparison.

**Prerequisite:** Pass the relevant gates from [week 19](week-19.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-134"></a>

## Day 134 — GEMM cost model

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I37 — Module 37 — GEMM and Tensor Cores](../originals/inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md). Focus on gemm cost model, using the task below to select the relevant subsection.

**External entry:** [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). See [resource verification notes](../RESOURCES.md).

**Do:** Derive FLOPs and bytes for selected decode-like and prefill-like shapes.

**Done when:** State which dimensions represent batch/tokens, input features and outputs.

**Career block:** Recall one coding or debugging pattern relevant to optimization tradeoffs; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-134.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-135"></a>

## Day 135 — Tiled multiplication

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I38 — Module 38 — Tiling, Pipelining, and Fusion](../originals/inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md). Focus on tiled multiplication, using the task below to select the relevant subsection.

**External entry:** [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). See [resource verification notes](../RESOURCES.md).

**Do:** Implement a small tiled CUDA or Triton GEMM reusing existing kernel skills.

**Done when:** Odd valid dimensions agree with the trusted library result.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to optimization tradeoffs. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-135.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-136"></a>

## Day 136 — Fused epilogue

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I38 — Module 38 — Tiling, Pipelining, and Fusion](../originals/inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md). Focus on fused epilogue, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide: pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html). See [resource verification notes](../RESOURCES.md).

**Do:** Add one bias or activation epilogue and compare against separate operations.

**Done when:** The full result matches the same numerical oracle.

**Career block:** Add one verified optimization tradeoffs observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-136.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-137"></a>

## Day 137 — Library baseline

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I37 — Module 37 — GEMM and Tensor Cores](../originals/inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on library baseline, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). See [resource verification notes](../RESOURCES.md).

**Do:** Compare the educational kernel with the installed optimized library under identical conditions.

**Done when:** Report absolute latency, error and any shapes where your kernel is slower.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-137.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-138"></a>

## Day 138 — CUTLASS/CuTe source tour

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I40 — Module 40 — CUTLASS and CuTe](../originals/inference-engineering-curriculum/modules/40-cutlass-and-cute.md). Focus on cutlass/cute source tour, using the task below to select the relevant subsection.

**External entry:** [CUTLASS overview and profiler](https://docs.nvidia.com/cutlass/latest/overview.html). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect one supported example and trace its shape/layout configuration; build only if the environment is ready.

**Done when:** Explain one layout or tile choice; full library proficiency is deferred.

**Career block:** Explain optimization tradeoffs aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-138.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-139"></a>

## Day 139 — Optimization decision

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [I37 — Module 37 — GEMM and Tensor Cores](../originals/inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md); [I38 — Module 38 — Tiling, Pipelining, and Fusion](../originals/inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md). Focus on optimization decision, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). See [resource verification notes](../RESOURCES.md).

**Do:** Choose one configuration from measured shape coverage and document why other variants lose.

**Done when:** A claimed speedup exceeds observed variation and includes end-to-end costs.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use optimization tradeoffs as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-139.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain reuse, occupancy constraints and honest baseline selection. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-140"></a>

## Day 140 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain reuse, occupancy constraints and honest baseline selection.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
