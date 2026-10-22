# Week 10: Quality evaluation and a stronger serving report

Days 064–070 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Held-out evaluation fixtures and benchmark report v2.

**Prerequisite:** Pass the relevant gates from [week 09](week-09.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-064"></a>

## Day 064 — Quality contract

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md); [I19 — Module 19 — Calibration, Error Metrics, and Evaluation](../originals/inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md). Focus on quality contract, using the task below to select the relevant subsection.

**External entry:** [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html). See [resource verification notes](../RESOURCES.md).

**Do:** Create a small reviewed evaluation set with answerable and unanswerable synthetic examples and a held-out split.

**Done when:** Define scoring and denominators before trying optimization.

**Career block:** Implement retry/backoff logic with a bounded total budget. Use the [career reference](../originals/ai-career-acceleration/weeks/week-10.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-064.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-065"></a>

## Day 065 — Generation evaluation

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on generation evaluation, using the task below to select the relevant subsection.

**External entry:** [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html). See [resource verification notes](../RESOURCES.md).

**Do:** Run the reference model or selected serving model on fixed fixtures; save outputs and configuration.

**Done when:** Report failures separately and inspect at least two disagreement cases.

**Career block:** Act on one relevant application or response. Use the [career reference](../originals/ai-career-acceleration/weeks/week-10.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-065.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-066"></a>

## Day 066 — Latency distributions

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md); [P18 — Module 18 — PyTorch Profiler, Memory, and Diagnostics](../originals/pytorch-systems-and-deployment-curriculum/modules/18-pytorch-profiler-memory-and-diagnostics.md). Focus on latency distributions, using the task below to select the relevant subsection.

**External entry:** [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/). See [resource verification notes](../RESOURCES.md).

**Do:** Extend the runner to report sample counts, TTFT, completion latency and output throughput with explicit timing boundaries.

**Done when:** A delayed-first-output fixture produces the expected recorded timing.

**Career block:** Draft the reliability lesson from a timeout/cancellation test. Use the [career reference](../originals/ai-career-acceleration/weeks/week-10.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-066.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-067"></a>

## Day 067 — Memory hierarchy

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E11 — Week 11: Profiling and the memory hierarchy](../originals/edge-inference-roadmap/weeks/week-11.md); [I05 — Module 05 — Performance Modeling and Roofline](../originals/inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md). Focus on memory hierarchy, using the task below to select the relevant subsection.

**External entry:** [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp). See [resource verification notes](../RESOURCES.md).

**Do:** Profile the native operator or decoder; compare a predicted bottleneck with observed allocations or timing.

**Done when:** Identify one concrete optimization hypothesis and a falsifying test.

**Career block:** Rehearse a small take-home task; stop at the time limit. Use the [career reference](../originals/ai-career-acceleration/weeks/week-10.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-067.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-068"></a>

## Day 068 — Contribution scouting

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P60 — Module 60 — PyTorch Testing, CI, and Triage](../originals/pytorch-systems-and-deployment-curriculum/modules/60-pytorch-testing-ci-and-triage.md); [I46 — Module 46 — vLLM Internals and Contributions](../originals/inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md). Focus on contribution scouting, using the task below to select the relevant subsection.

**External entry:** [PyTorch contributing guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md). See [resource verification notes](../RESOURCES.md).

**Do:** Reproduce one small issue or documentation ambiguity in the primary runtime; save a local minimal reproducer.

**Done when:** Confirm behavior at a pinned revision; do not invent an issue to satisfy a quota.

**Career block:** Explain idempotency, cancellation and failure accounting. Use the [career reference](../originals/ai-career-acceleration/weeks/week-10.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-068.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-069"></a>

## Day 069 — Evidence release

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on evidence release, using the task below to select the relevant subsection.

**External entry:** [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html). See [resource verification notes](../RESOURCES.md).

**Do:** Package the serving report and one quality baseline alongside the decoder tests.

**Done when:** All later optimizations have a stable correctness and quality baseline.

**Career block:** Review the take-home, rehearse system design and update the pipeline. Use the [career reference](../originals/ai-career-acceleration/weeks/week-10.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-069.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Distinguish operator error, model quality and serving performance. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-070"></a>

## Day 070 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Distinguish operator error, model quality and serving performance.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
