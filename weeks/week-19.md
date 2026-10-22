# Week 19: CUDA programming and memory movement

Days 127–133 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Native vector/reduction kernels with checked launch bounds.

**Prerequisite:** Pass the relevant gates from [week 18](week-18.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-127"></a>

## Day 127 — Device inventory

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I35 — Module 35 — GPU Architecture and Blackwell](../originals/inference-engineering-curriculum/modules/35-gpu-architecture-and-blackwell.md); [I36 — Module 36 — CUDA Execution, Memory, and Streams](../originals/inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md). Focus on device inventory, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). See [resource verification notes](../RESOURCES.md).

**Do:** Query the actual device and read the matching CUDA programming guide; map grid/block/thread indices.

**Done when:** Record device properties rather than assuming all GPUs in one family are equivalent.

**Career block:** Recall one coding or debugging pattern relevant to CUDA debugging; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-127.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-128"></a>

## Day 128 — Vector kernel

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I36 — Module 36 — CUDA Execution, Memory, and Streams](../originals/inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md). Focus on vector kernel, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). See [resource verification notes](../RESOURCES.md).

**Do:** Write a CUDA vector operation with bounds checks and explicit host/device transfers.

**Done when:** Zero/one/tail-length cases agree with the CPU oracle.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to CUDA debugging. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-128.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-129"></a>

## Day 129 — Memory access

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I36 — Module 36 — CUDA Execution, Memory, and Streams](../originals/inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md); [P09 — Module 09 — Devices, Dtypes, Memory, and Streams](../originals/pytorch-systems-and-deployment-curriculum/modules/09-devices-dtypes-memory-and-streams.md). Focus on memory access, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). See [resource verification notes](../RESOURCES.md).

**Do:** Compare a coalesced and deliberately strided access pattern on supported hardware.

**Done when:** Explain observed timing with bytes moved and access pattern.

**Career block:** Add one verified CUDA debugging observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-129.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-130"></a>

## Day 130 — Shared reduction

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I36 — Module 36 — CUDA Execution, Memory, and Streams](../originals/inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md). Focus on shared reduction, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). See [resource verification notes](../RESOURCES.md).

**Do:** Implement one small block reduction with explicit synchronization.

**Done when:** Tail blocks and more than one block size pass the same oracle.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-130.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-131"></a>

## Day 131 — CUDA extension

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P36 — Module 36 — C++/CUDA Custom Operators](../originals/pytorch-systems-and-deployment-curriculum/modules/36-cpp-cuda-custom-operators.md); [I41 — Module 41 — PyTorch Custom Operators](../originals/inference-engineering-curriculum/modules/41-pytorch-custom-operators.md). Focus on cuda extension, using the task below to select the relevant subsection.

**External entry:** [Custom C++ and CUDA operators tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html). See [resource verification notes](../RESOURCES.md).

**Do:** Add the verified kernel to the custom-op package and retain the CPU fallback.

**Done when:** Device dispatch and unsupported-input handling behave as specified.

**Career block:** Explain CUDA debugging aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-131.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-132"></a>

## Day 132 — CUDA gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [I36 — Module 36 — CUDA Execution, Memory, and Streams](../originals/inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on cuda gate, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). See [resource verification notes](../RESOURCES.md).

**Do:** Use the available sanitizer/profiler on the smallest kernel and write a synchronization note.

**Done when:** No unexplained numerical mismatch or invalid memory access remains.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use CUDA debugging as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-132.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain threads, blocks, memory spaces and synchronization. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-133"></a>

## Day 133 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain threads, blocks, memory spaces and synchronization.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
