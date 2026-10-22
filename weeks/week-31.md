# Week 31: Edge lifecycle, sustained load and optional accelerator survey

Days 211–217 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Resource-policy tests and device/host comparison.

**Prerequisite:** Pass the relevant gates from [week 30](week-30.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-211"></a>

## Day 211 — Lifecycle

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E17 — Week 17: Lifecycle, sustained performance and resource policies](../originals/edge-inference-roadmap/weeks/week-17.md). Focus on lifecycle, using the task below to select the relevant subsection.

**External entry:** [Android application fundamentals](https://developer.android.com/guide/components/fundamentals). See [resource verification notes](../RESOURCES.md).

**Do:** Test stop/restart, backgrounding and request cancellation for the minimal app or host service.

**Done when:** Each request gets one terminal outcome and released resources.

**Career block:** Recall one coding or debugging pattern relevant to resource policies; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-211.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-212"></a>

## Day 212 — Sustained workload

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E17 — Week 17: Lifecycle, sustained performance and resource policies](../originals/edge-inference-roadmap/weeks/week-17.md); [E14 — Week 14: ARM NEON, threading and measured optimization](../originals/edge-inference-roadmap/weeks/week-14.md). Focus on sustained workload, using the task below to select the relevant subsection.

**External entry:** [Android application fundamentals](https://developer.android.com/guide/components/fundamentals). See [resource verification notes](../RESOURCES.md).

**Do:** Run repeated bounded requests and record latency, memory and available thermal information.

**Done when:** Separate warm-device behavior from a single cool-device result.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to resource policies. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-212.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-213"></a>

## Day 213 — Backend comparison

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E20 — Week 20: Backend comparison and inference infrastructure](../originals/edge-inference-roadmap/weeks/week-20.md). Focus on backend comparison, using the task below to select the relevant subsection.

**External entry:** [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md). See [resource verification notes](../RESOURCES.md).

**Do:** Compare two actually available paths under the same model/input contract, or compare one path with a documented blocked route.

**Done when:** Keep unsupported features explicit rather than filling missing measurements.

**Career block:** Add one verified resource policies observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-213.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-214"></a>

## Day 214 — Vendor constraints

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E19 — Week 19: Qualcomm NPU deployment and vendor runtime constraints](../originals/edge-inference-roadmap/weeks/week-19.md). Focus on vendor constraints, using the task below to select the relevant subsection.

**External entry:** [Qualcomm Genie LLM tutorial](https://github.com/qualcomm/ai-hub-apps/blob/release/tutorials/llm_on_genie/README.md). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect the Qualcomm/backend guide only if the device is relevant; otherwise write a compatibility survey.

**Done when:** Check SoC, SDK, firmware, model and partition support before claiming NPU execution.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-214.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-215"></a>

## Day 215 — Offline document slice

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E21 — Week 21: Offline document assistant: retrieval and resource ownership](../originals/edge-inference-roadmap/weeks/week-21.md); [E22 — Week 22: Capstone: memory management, quality and optional multimodality](../originals/edge-inference-roadmap/weeks/week-22.md). Focus on offline document slice, using the task below to select the relevant subsection.

**External entry:** [SQLite documentation](https://www.sqlite.org/docs.html). See [resource verification notes](../RESOURCES.md).

**Do:** Add a tiny public-text corpus with exact retrieval and citations to the existing app/host path.

**Done when:** One answerable and one unanswerable question produce the intended behavior.

**Career block:** Explain resource policies aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-215.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-216"></a>

## Day 216 — Edge quality gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [E23 — Week 23: Evaluation, regression testing and sustained application behavior](../originals/edge-inference-roadmap/weeks/week-23.md); [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md). Focus on edge quality gate, using the task below to select the relevant subsection.

**External entry:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). See [resource verification notes](../RESOURCES.md).

**Do:** Evaluate a small held-out fixture set and test a missing/corrupt model asset.

**Done when:** Label host-precomputed embeddings and the achieved on-device level precisely.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use resource policies as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-216.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Handle cancellation, restart and limited memory without false accelerator claims. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-217"></a>

## Day 217 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Handle cancellation, restart and limited memory without false accelerator claims.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
