# Week 35: One upstream-quality contribution

Days 239–245 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Minimal patch, regression test and contribution dossier.

**Prerequisite:** Pass the relevant gates from [week 34](week-34.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-239"></a>

## Day 239 — Contribution scope

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P59 — Module 59 — PyTorch Source Tree and Build](../originals/pytorch-systems-and-deployment-curriculum/modules/59-pytorch-source-tree-and-build.md); [P61 — Module 61 — PyTorch Contribution Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/61-pytorch-contribution-engineering.md). Focus on contribution scope, using the task below to select the relevant subsection.

**External entry:** [PyTorch from-source build](https://github.com/pytorch/pytorch#from-source). See [resource verification notes](../RESOURCES.md).

**Do:** Select the already-reproduced issue in the primary runtime or PyTorch ecosystem; reread current contribution rules.

**Done when:** Write expected behavior, observed behavior and the smallest justified fix.

**Career block:** Recall one coding or debugging pattern relevant to contribution review; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-239.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-240"></a>

## Day 240 — Failing regression

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P60 — Module 60 — PyTorch Testing, CI, and Triage](../originals/pytorch-systems-and-deployment-curriculum/modules/60-pytorch-testing-ci-and-triage.md); [I46 — Module 46 — vLLM Internals and Contributions](../originals/inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md). Focus on failing regression, using the task below to select the relevant subsection.

**External entry:** [PyTorch contributing guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md). See [resource verification notes](../RESOURCES.md).

**Do:** Create a minimal upstream-style test or documentation reproduction for the issue.

**Done when:** Demonstrate the failure on the chosen base revision.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to contribution review. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-240.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-241"></a>

## Day 241 — Small patch

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P61 — Module 61 — PyTorch Contribution Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/61-pytorch-contribution-engineering.md). Focus on small patch, using the task below to select the relevant subsection.

**External entry:** [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md). See [resource verification notes](../RESOURCES.md).

**Do:** Implement the narrow change without unrelated cleanup.

**Done when:** The regression passes and the diff matches the stated problem.

**Career block:** Add one verified contribution review observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-241.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-242"></a>

## Day 242 — Focused validation

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P60 — Module 60 — PyTorch Testing, CI, and Triage](../originals/pytorch-systems-and-deployment-curriculum/modules/60-pytorch-testing-ci-and-triage.md). Focus on focused validation, using the task below to select the relevant subsection.

**External entry:** [PyTorch contributing guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md). See [resource verification notes](../RESOURCES.md).

**Do:** Run the project's required relevant checks and inspect compatibility with adjacent behavior.

**Done when:** Record commands, results and unavailable checks honestly.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-242.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-243"></a>

## Day 243 — Reviewer explanation

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P61 — Module 61 — PyTorch Contribution Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/61-pytorch-contribution-engineering.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on reviewer explanation, using the task below to select the relevant subsection.

**External entry:** [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md). See [resource verification notes](../RESOURCES.md).

**Do:** Prepare a PR description with the trigger, before/after behavior and validation; include benchmarks only for performance claims.

**Done when:** Every claim can be traced to a test, record or reproducible command.

**Career block:** Explain contribution review aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-243.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-244"></a>

## Day 244 — Contribution gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P61 — Module 61 — PyTorch Contribution Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/61-pytorch-contribution-engineering.md). Focus on contribution gate, using the task below to select the relevant subsection.

**External entry:** [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md). See [resource verification notes](../RESOURCES.md).

**Do:** Review the patch as a maintainer would; submit manually only if appropriate and desired, or retain an upstream-ready dossier.

**Done when:** Completion depends on patch quality and evidence, not on a maintainer merging it.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use contribution review as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-244.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: The change solves a reproduced issue and follows the target project's current rules. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-245"></a>

## Day 245 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** The change solves a reproduced issue and follows the target project's current rules.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
