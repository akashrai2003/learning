# Week 34: Integrate the flagship

Days 232–238 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Reproducible benchmark/reliability lab with one user-facing slice.

**Prerequisite:** Pass the relevant gates from [week 33](week-33.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-232"></a>

## Day 232 — Integration contract

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P62 — Module 62 — PyTorch Ecosystem Capstone](../originals/pytorch-systems-and-deployment-curriculum/modules/62-pytorch-ecosystem-capstone.md); [E21 — Week 21: Offline document assistant: retrieval and resource ownership](../originals/edge-inference-roadmap/weeks/week-21.md). Focus on integration contract, using the task below to select the relevant subsection.

**External entry:** [PyTorch ecosystem](https://pytorch.org/ecosystem/). See [resource verification notes](../RESOURCES.md).

**Do:** Connect the common runner, model/runtime configuration and optional document retrieval slice.

**Done when:** Define stable request, result and artifact schemas.

**Career block:** Recall one coding or debugging pattern relevant to capstone integration; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-232.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-233"></a>

## Day 233 — Asset integrity

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E22 — Week 22: Capstone: memory management, quality and optional multimodality](../originals/edge-inference-roadmap/weeks/week-22.md); [P16 — Module 16 — Serialization, Checkpoints, and Model Artifacts](../originals/pytorch-systems-and-deployment-curriculum/modules/16-serialization-checkpoints-and-model-artifacts.md). Focus on asset integrity, using the task below to select the relevant subsection.

**External entry:** [SQLite documentation](https://www.sqlite.org/docs.html). See [resource verification notes](../RESOURCES.md).

**Do:** Add validated manifests and bounded loading for the selected assets and index.

**Done when:** Missing or corrupt assets fail before producing a misleading answer.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to capstone integration. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-233.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-234"></a>

## Day 234 — Evidence and quality

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E21 — Week 21: Offline document assistant: retrieval and resource ownership](../originals/edge-inference-roadmap/weeks/week-21.md); [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md). Focus on evidence and quality, using the task below to select the relevant subsection.

**External entry:** [SQLite documentation](https://www.sqlite.org/docs.html). See [resource verification notes](../RESOURCES.md).

**Do:** Use reviewed questions with source references and an abstention policy for the document slice, or equivalent task fixtures for serving.

**Done when:** Keep held-out cases separate from tuning inputs.

**Career block:** Add one verified capstone integration observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-234.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-235"></a>

## Day 235 — Reliability

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E23 — Week 23: Evaluation, regression testing and sustained application behavior](../originals/edge-inference-roadmap/weeks/week-23.md); [I34 — Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache](../originals/inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md). Focus on reliability, using the task below to select the relevant subsection.

**External entry:** [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). See [resource verification notes](../RESOURCES.md).

**Do:** Test queue bounds, cancellation, partial failure and clean restart on the primary path.

**Done when:** No request disappears from results or finishes twice.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-235.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-236"></a>

## Day 236 — End-to-end report

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md); [E23 — Week 23: Evaluation, regression testing and sustained application behavior](../originals/edge-inference-roadmap/weeks/week-23.md). Focus on end-to-end report, using the task below to select the relevant subsection.

**External entry:** [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/). See [resource verification notes](../RESOURCES.md).

**Do:** Run a small reproducible workload under a declared latency and memory budget.

**Done when:** Report quality and failures alongside latency, throughput and resource use.

**Career block:** Explain capstone integration aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-236.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-237"></a>

## Day 237 — Integration gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P62 — Module 62 — PyTorch Ecosystem Capstone](../originals/pytorch-systems-and-deployment-curriculum/modules/62-pytorch-ecosystem-capstone.md); [E24 — Week 24: Release engineering, portfolio and the next specialization](../originals/edge-inference-roadmap/weeks/week-24.md). Focus on integration gate, using the task below to select the relevant subsection.

**External entry:** [PyTorch ecosystem](https://pytorch.org/ecosystem/). See [resource verification notes](../RESOURCES.md).

**Do:** Ask a fresh shell to reproduce the smallest demo from the README.

**Done when:** Document the achieved scope; keep a native educational engine separate from production-runtime claims.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use capstone integration as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-237.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Another developer can exercise the main path and inspect failures. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-238"></a>

## Day 238 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Another developer can exercise the main path and inspect failures.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
