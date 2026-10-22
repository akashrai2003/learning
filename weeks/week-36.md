# Week 36: Buffer 3 — deployment and contribution repairs

Days 246–252 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Clean reproducibility, closed correctness blockers and final scope.

**Prerequisite:** Pass the relevant gates from [week 35](week-35.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-246"></a>

## Day 246 — Release triage

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P62 — Module 62 — PyTorch Ecosystem Capstone](../originals/pytorch-systems-and-deployment-curriculum/modules/62-pytorch-ecosystem-capstone.md); [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md). Focus on release triage, using the task below to select the relevant subsection.

**External entry:** [PyTorch ecosystem](https://pytorch.org/ecosystem/). See [resource verification notes](../RESOURCES.md).

**Do:** List capstone and contribution blockers; choose the two that affect correctness or reproducibility most.

**Done when:** Each blocker has a concrete failure and acceptance check.

**Career block:** Recall one coding or debugging pattern relevant to release readiness; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-246.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-247"></a>

## Day 247 — Artifact repair

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P51 — Module 51 — `torch.export` and Program Capture](../originals/pytorch-systems-and-deployment-curriculum/modules/51-torch-export-and-program-capture.md); [P53 — Module 53 — ONNX Runtime and Execution Providers](../originals/pytorch-systems-and-deployment-curriculum/modules/53-onnx-runtime-and-execution-providers.md). Focus on artifact repair, using the task below to select the relevant subsection.

**External entry:** [torch.export](https://docs.pytorch.org/docs/stable/export.html). See [resource verification notes](../RESOURCES.md).

**Do:** Re-export or reload the primary artifact in a clean directory and compare output contracts.

**Done when:** No dependency on a stale local cache is hidden.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to release readiness. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-247.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-248"></a>

## Day 248 — Reliability repair

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E17 — Week 17: Lifecycle, sustained performance and resource policies](../originals/edge-inference-roadmap/weeks/week-17.md); [E23 — Week 23: Evaluation, regression testing and sustained application behavior](../originals/edge-inference-roadmap/weeks/week-23.md). Focus on reliability repair, using the task below to select the relevant subsection.

**External entry:** [Android application fundamentals](https://developer.android.com/guide/components/fundamentals). See [resource verification notes](../RESOURCES.md).

**Do:** Repeat the weakest cancellation, restart or asset-integrity scenario.

**Done when:** State and request accounting remain correct after recovery.

**Career block:** Add one verified release readiness observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-248.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-249"></a>

## Day 249 — Patch repair

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P60 — Module 60 — PyTorch Testing, CI, and Triage](../originals/pytorch-systems-and-deployment-curriculum/modules/60-pytorch-testing-ci-and-triage.md); [P61 — Module 61 — PyTorch Contribution Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/61-pytorch-contribution-engineering.md). Focus on patch repair, using the task below to select the relevant subsection.

**External entry:** [PyTorch contributing guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md). See [resource verification notes](../RESOURCES.md).

**Do:** Address one review finding or perform a self-review if no external review exists.

**Done when:** Re-run the focused regression after the change.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-249.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-250"></a>

## Day 250 — Evidence audit

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md); [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md). Focus on evidence audit, using the task below to select the relevant subsection.

**External entry:** [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/). See [resource verification notes](../RESOURCES.md).

**Do:** Regenerate one key report and reconcile every metric with raw data.

**Done when:** Remove or narrow unsupported hardware, quality and performance claims.

**Career block:** Explain release readiness aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-250.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-251"></a>

## Day 251 — Final-scope review

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P62 — Module 62 — PyTorch Ecosystem Capstone](../originals/pytorch-systems-and-deployment-curriculum/modules/62-pytorch-ecosystem-capstone.md). Focus on final-scope review, using the task below to select the relevant subsection.

**External entry:** [PyTorch ecosystem](https://pytorch.org/ecosystem/). See [resource verification notes](../RESOURCES.md).

**Do:** Use the final gate checklist and freeze new features for the remaining three weeks.

**Done when:** Unfinished stretch work moves to EXTENSIONS.md rather than consuming release time.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use release readiness as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-251.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Resolve release-blocking gaps before polishing presentation. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-252"></a>

## Day 252 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Resolve release-blocking gaps before polishing presentation.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
