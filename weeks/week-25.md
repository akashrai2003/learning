# Week 25: A production runtime: vLLM as the default

Days 169–175 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Request-to-kernel source map and a real benchmark.

**Prerequisite:** Pass the relevant gates from [week 24](week-24.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-169"></a>

## Day 169 — Request path

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I46 — Module 46 — vLLM Internals and Contributions](../originals/inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md). Focus on request path, using the task below to select the relevant subsection.

**External entry:** [vLLM documentation and developer guide](https://docs.vllm.ai/en/latest/). See [resource verification notes](../RESOURCES.md).

**Do:** Trace one request through the pinned vLLM checkout or your chosen primary runtime.

**Done when:** Record entry, scheduling, model execution, cache and output boundaries.

**Career block:** Recall one coding or debugging pattern relevant to runtime ownership; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-169.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-170"></a>

## Day 170 — Cache lifecycle

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I46 — Module 46 — vLLM Internals and Contributions](../originals/inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md); [I31 — Module 31 — PagedAttention and KV Memory Management](../originals/inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md). Focus on cache lifecycle, using the task below to select the relevant subsection.

**External entry:** [vLLM documentation and developer guide](https://docs.vllm.ai/en/latest/). See [resource verification notes](../RESOURCES.md).

**Do:** Find cache allocation and request completion code plus one relevant test.

**Done when:** Explain where ownership and reclamation are enforced.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to runtime ownership. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-170.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-171"></a>

## Day 171 — Real workload

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I46 — Module 46 — vLLM Internals and Contributions](../originals/inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md); [I34 — Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache](../originals/inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md). Focus on real workload, using the task below to select the relevant subsection.

**External entry:** [vLLM documentation and developer guide](https://docs.vllm.ai/en/latest/). See [resource verification notes](../RESOURCES.md).

**Do:** Run the existing benchmark harness with fixed arrivals and bounded concurrency.

**Done when:** Record rejected and timed-out requests rather than discarding them.

**Career block:** Add one verified runtime ownership observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-171.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-172"></a>

## Day 172 — Operating point

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md); [I46 — Module 46 — vLLM Internals and Contributions](../originals/inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md). Focus on operating point, using the task below to select the relevant subsection.

**External entry:** [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/). See [resource verification notes](../RESOURCES.md).

**Do:** Sweep a small number of request rates and compare latency with throughput.

**Done when:** Select a useful operating point from stated latency/quality requirements.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-172.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-173"></a>

## Day 173 — Minimal reproduction

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I46 — Module 46 — vLLM Internals and Contributions](../originals/inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md); [P60 — Module 60 — PyTorch Testing, CI, and Triage](../originals/pytorch-systems-and-deployment-curriculum/modules/60-pytorch-testing-ci-and-triage.md). Focus on minimal reproduction, using the task below to select the relevant subsection.

**External entry:** [vLLM documentation and developer guide](https://docs.vllm.ai/en/latest/). See [resource verification notes](../RESOURCES.md).

**Do:** Reduce one observed anomaly or confusing documented behavior to a small fixture/script.

**Done when:** Separate configuration mistakes from a reproducible runtime issue.

**Career block:** Explain runtime ownership aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-173.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-174"></a>

## Day 174 — Runtime dossier

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [I46 — Module 46 — vLLM Internals and Contributions](../originals/inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md). Focus on runtime dossier, using the task below to select the relevant subsection.

**External entry:** [vLLM documentation and developer guide](https://docs.vllm.ai/en/latest/). See [resource verification notes](../RESOURCES.md).

**Do:** Connect one measured behavior to a source path and document the limits of that explanation.

**Done when:** A reviewer can reproduce the result at the recorded revision.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use runtime ownership as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-174.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Relate the simulator to actual behavior without assuming identical internals. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-175"></a>

## Day 175 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Relate the simulator to actual behavior without assuming identical internals.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
