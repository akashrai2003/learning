# Week 38: Portfolio, interviews and systems explanations

Days 260–266 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Three concise case studies and a skills evidence map.

**Prerequisite:** Pass the relevant gates from [week 37](week-37.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-260"></a>

## Day 260 — Case study one

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on case study one, using the task below to select the relevant subsection.

**External entry:** [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html). See [resource verification notes](../RESOURCES.md).

**Do:** Write the baseline-to-optimization story with question, method, result and limitations.

**Done when:** Every number links to a raw result or is labeled an estimate.

**Career block:** Recall one coding or debugging pattern relevant to portfolio explanations; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-260.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-261"></a>

## Day 261 — Case study two

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I12 — Module 12 — MQA, GQA, and the KV Cache](../originals/inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md); [P21 — Module 21 — Attention, KV Cache, and the PyTorch Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/21-attention-kv-cache-and-pytorch-bridge.md). Focus on case study two, using the task below to select the relevant subsection.

**External entry:** [“Fast Transformer Decoding: One Write-Head is All You Need”](https://arxiv.org/abs/1911.02150). See [resource verification notes](../RESOURCES.md).

**Do:** Explain cached decoding and one bug you repaired using shapes and state diagrams on paper.

**Done when:** Reconstruct the critical function without notes.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to portfolio explanations. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-261.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-262"></a>

## Day 262 — Case study three

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P51 — Module 51 — `torch.export` and Program Capture](../originals/pytorch-systems-and-deployment-curriculum/modules/51-torch-export-and-program-capture.md); [E20 — Week 20: Backend comparison and inference infrastructure](../originals/edge-inference-roadmap/weeks/week-20.md). Focus on case study three, using the task below to select the relevant subsection.

**External entry:** [torch.export](https://docs.pytorch.org/docs/stable/export.html). See [resource verification notes](../RESOURCES.md).

**Do:** Explain your deployment decision and resource/reliability tradeoffs.

**Done when:** Distinguish tested paths from the alternatives you only surveyed.

**Career block:** Add one verified portfolio explanations observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-262.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-263"></a>

## Day 263 — System-design mock

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I45 — Module 45 — Distributed Inference Parallelism](../originals/inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md); [P50 — Module 50 — Training–Serving Parallelism Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/50-training-serving-parallelism-bridge.md). Focus on system-design mock, using the task below to select the relevant subsection.

**External entry:** [Megatron-LM paper](https://arxiv.org/abs/1909.08053). See [resource verification notes](../RESOURCES.md).

**Do:** Design a bounded inference service under explicit latency, memory and failure constraints.

**Done when:** Discuss overload, quality, observability and recovery as well as GPU utilization.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-263.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-264"></a>

## Day 264 — Debugging mock

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P42 — Module 42 — Compiler Debugging, Minimization, and Regression Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/42-compiler-debugging-minimization-and-regression.md); [P60 — Module 60 — PyTorch Testing, CI, and Triage](../originals/pytorch-systems-and-deployment-curriculum/modules/60-pytorch-testing-ci-and-triage.md). Focus on debugging mock, using the task below to select the relevant subsection.

**External entry:** [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html). See [resource verification notes](../RESOURCES.md).

**Do:** Take a saved compiler, numerics or runtime failure and debug it under a time limit.

**Done when:** Show hypothesis, smallest reproduction and verification of the repair.

**Career block:** Explain portfolio explanations aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-264.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-265"></a>

## Day 265 — Portfolio gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P62 — Module 62 — PyTorch Ecosystem Capstone](../originals/pytorch-systems-and-deployment-curriculum/modules/62-pytorch-ecosystem-capstone.md). Focus on portfolio gate, using the task below to select the relevant subsection.

**External entry:** [PyTorch ecosystem](https://pytorch.org/ecosystem/). See [resource verification notes](../RESOURCES.md).

**Do:** Link the three cases, selected code and contribution dossier from a concise portfolio index.

**Done when:** Skills claimed in the resume have demonstrable evidence; no checklist-based mastery claim.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use portfolio explanations as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-265.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Defend decisions without relying on prepared slides. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-266"></a>

## Day 266 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Defend decisions without relying on prepared slides.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
