# Week 17: Triton fundamentals and one fused operator

Days 113–119 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Masked vector kernel and a fused softmax or norm.

**Prerequisite:** Pass the relevant gates from [week 16](week-16.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-113"></a>

## Day 113 — Execution model

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I39 — Module 39 — Triton Kernels](../originals/inference-engineering-curriculum/modules/39-triton-kernels.md); [I35 — Module 35 — GPU Architecture and Blackwell](../originals/inference-engineering-curriculum/modules/35-gpu-architecture-and-blackwell.md); [P34 — Module 34 — Triton with PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/34-triton-with-pytorch.md). Focus on execution model, using the task below to select the relevant subsection.

**External entry:** [Triton vector addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html). See [resource verification notes](../RESOURCES.md).

**Do:** Read Triton vector addition and map program IDs, blocks, loads and masks onto a small kernel.

**Done when:** Odd input lengths are correct and out-of-range loads are masked.

**Career block:** Recall one coding or debugging pattern relevant to kernel evidence; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-113.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-114"></a>

## Day 114 — Reduction kernel

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I39 — Module 39 — Triton Kernels](../originals/inference-engineering-curriculum/modules/39-triton-kernels.md); [P34 — Module 34 — Triton with PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/34-triton-with-pytorch.md). Focus on reduction kernel, using the task below to select the relevant subsection.

**External entry:** [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html). See [resource verification notes](../RESOURCES.md).

**Do:** Implement a small row softmax or normalization kernel using a reference oracle.

**Done when:** Large-magnitude values and non-power-of-two widths pass tolerance checks.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to kernel evidence. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-114.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-115"></a>

## Day 115 — Fusion hypothesis

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I05 — Module 05 — Performance Modeling and Roofline](../originals/inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md); [I38 — Module 38 — Tiling, Pipelining, and Fusion](../originals/inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md). Focus on fusion hypothesis, using the task below to select the relevant subsection.

**External entry:** [Roofline: An Insightful Visual Performance Model](https://digicoll.lib.berkeley.edu/record/136692/files/EECS-2008-134.pdf). See [resource verification notes](../RESOURCES.md).

**Do:** Compare separate operations with the fused path and estimate bytes avoided.

**Done when:** Include launch and layout conversion costs in the comparison.

**Career block:** Add one verified kernel evidence observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-115.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-116"></a>

## Day 116 — Shape sweep

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P34 — Module 34 — Triton with PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/34-triton-with-pytorch.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on shape sweep, using the task below to select the relevant subsection.

**External entry:** [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Measure a few small and large shapes after warm-up; keep raw timings and variance.

**Done when:** Report a slow case as well as any speedup.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-116.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-117"></a>

## Day 117 — Compiler integration

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P33 — Module 33 — TorchInductor: Fusion, Scheduling, and Code Generation](../originals/pytorch-systems-and-deployment-curriculum/modules/33-torchinductor-fusion-scheduling-and-codegen.md); [I39 — Module 39 — Triton Kernels](../originals/inference-engineering-curriculum/modules/39-triton-kernels.md). Focus on compiler integration, using the task below to select the relevant subsection.

**External entry:** [TorchInductor docs](https://docs.pytorch.org/docs/stable/torch.compiler.html). See [resource verification notes](../RESOURCES.md).

**Do:** Call the kernel from a small PyTorch module and inspect the supported compile integration.

**Done when:** Eager and compiled results agree or the unsupported path is documented.

**Career block:** Explain kernel evidence aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-117.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-118"></a>

## Day 118 — Kernel gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P34 — Module 34 — Triton with PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/34-triton-with-pytorch.md); [I39 — Module 39 — Triton Kernels](../originals/inference-engineering-curriculum/modules/39-triton-kernels.md). Focus on kernel gate, using the task below to select the relevant subsection.

**External entry:** [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Reconstruct the tail mask and reduction logic without copying the tutorial.

**Done when:** The report ties a measured effect to memory traffic or launch overhead.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use kernel evidence as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-118.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Match the oracle on tails and justify measured performance. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-119"></a>

## Day 119 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Match the oracle on tails and justify measured performance.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
