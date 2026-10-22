# Week 23: Cache allocation and serving schedules

Days 155–161 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Deterministic paged-cache and batching simulator.

**Prerequisite:** Pass the relevant gates from [week 22](week-22.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-155"></a>

## Day 155 — Paged-cache model

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I31 — Module 31 — PagedAttention and KV Memory Management](../originals/inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md). Focus on paged-cache model, using the task below to select the relevant subsection.

**External entry:** [Efficient Memory Management with PagedAttention](https://arxiv.org/abs/2309.06180). See [resource verification notes](../RESOURCES.md).

**Do:** Implement a small block allocator simulation with request ownership and reference counts.

**Done when:** Allocate/free/reset sequences preserve invariants and reclaim all blocks.

**Career block:** Recall one coding or debugging pattern relevant to serving fairness; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-155.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-156"></a>

## Day 156 — Continuous batching

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I34 — Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache](../originals/inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md). Focus on continuous batching, using the task below to select the relevant subsection.

**External entry:** [vLLM optimization and chunked-prefill guide](https://docs.vllm.ai/en/latest/configuration/optimization/). See [resource verification notes](../RESOURCES.md).

**Do:** Add a discrete-time scheduler over fixed synthetic arrivals and token costs.

**Done when:** Requests finish exactly once and completed cache blocks are released.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to serving fairness. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-156.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-157"></a>

## Day 157 — Prefill and fairness

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I34 — Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache](../originals/inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md). Focus on prefill and fairness, using the task below to select the relevant subsection.

**External entry:** [vLLM optimization and chunked-prefill guide](https://docs.vllm.ai/en/latest/configuration/optimization/). See [resource verification notes](../RESOURCES.md).

**Do:** Add a bounded prefill budget and compare waiting-time behavior.

**Done when:** Test a long prompt mixed with short requests and record starvation risk.

**Career block:** Add one verified serving fairness observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-157.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-158"></a>

## Day 158 — Prefix reuse

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I31 — Module 31 — PagedAttention and KV Memory Management](../originals/inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md); [I34 — Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache](../originals/inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md). Focus on prefix reuse, using the task below to select the relevant subsection.

**External entry:** [Efficient Memory Management with PagedAttention](https://arxiv.org/abs/2309.06180). See [resource verification notes](../RESOURCES.md).

**Do:** Simulate shared-prefix blocks with explicit ownership and eviction rules.

**Done when:** Eviction cannot free blocks still referenced by another request.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-158.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-159"></a>

## Day 159 — Speculation survey

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I33 — Module 33 — Speculative Decoding](../originals/inference-engineering-curriculum/modules/33-speculative-decoding.md). Focus on speculation survey, using the task below to select the relevant subsection.

**External entry:** [Accelerating Large Language Model Decoding with Speculative Sampling](https://proceedings.mlr.press/v202/leviathan23a.html). See [resource verification notes](../RESOURCES.md).

**Do:** Work through a tiny acceptance example and model proposal/verification cost.

**Done when:** Distinguish a cost simulation from distribution-correct speculative sampling.

**Career block:** Explain serving fairness aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-159.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-160"></a>

## Day 160 — Scheduler report

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [I32 — Module 32 — Decoding, Sampling, and Logits Processing](../originals/inference-engineering-curriculum/modules/32-decoding-sampling-and-logits-processing.md); [I34 — Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache](../originals/inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md). Focus on scheduler report, using the task below to select the relevant subsection.

**External entry:** [The Curious Case of Neural Text Degeneration / nucleus sampling](https://arxiv.org/abs/1904.09751). See [resource verification notes](../RESOURCES.md).

**Do:** Report TTFT, completion latency, fragmentation and failures on one fixed arrival trace.

**Done when:** Label all results simulated and make no real-runtime throughput claim.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use serving fairness as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-160.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain fairness, cache reclamation and request-level metrics. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-161"></a>

## Day 161 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain fairness, cache reclamation and request-level metrics.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
