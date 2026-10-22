# Week 29: ExecuTorch export and a host edge baseline

Days 197–203 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Small ExecuTorch artifact and delegation report.

**Prerequisite:** Pass the relevant gates from [week 28](week-28.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-197"></a>

## Day 197 — Edge architecture

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P57 — Module 57 — ExecuTorch Architecture, EXIR, and Backends](../originals/pytorch-systems-and-deployment-curriculum/modules/57-executorch-architecture-exir-and-backends.md); [E18 — Week 18: ExecuTorch export, graph lowering and delegation](../originals/edge-inference-roadmap/weeks/week-18.md). Focus on edge architecture, using the task below to select the relevant subsection.

**External entry:** [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Trace export, lowering, partitioning and runtime execution for one official supported example.

**Done when:** Draw artifact boundaries and pin exporter/runtime versions.

**Career block:** Recall one coding or debugging pattern relevant to edge lowering; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-197.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-198"></a>

## Day 198 — Host execution

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P57 — Module 57 — ExecuTorch Architecture, EXIR, and Backends](../originals/pytorch-systems-and-deployment-curriculum/modules/57-executorch-architecture-exir-and-backends.md). Focus on host execution, using the task below to select the relevant subsection.

**External entry:** [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Export and run the smallest supported example on the host before targeting a phone.

**Done when:** Compare with eager output and preserve the commands.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to edge lowering. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-198.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-199"></a>

## Day 199 — Edge quantization

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P58 — Module 58 — ExecuTorch Quantization, Delegates, and Edge Deployment](../originals/pytorch-systems-and-deployment-curriculum/modules/58-executorch-quantization-delegates-and-edge.md); [P24 — Module 24 — TorchAO: Quantization, Sparsity, and Low-Bit Tensors](../originals/pytorch-systems-and-deployment-curriculum/modules/24-torchao-quantization-sparsity-and-low-bit.md). Focus on edge quantization, using the task below to select the relevant subsection.

**External entry:** [ExecuTorch backend and quantization guides](https://docs.pytorch.org/executorch/stable/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Apply a supported quantization configuration to the small example or inspect its exact constraints.

**Done when:** Quality and actual backend support are measured or explicitly marked blocked.

**Career block:** Add one verified edge lowering observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-199.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-200"></a>

## Day 200 — Delegate partition

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P58 — Module 58 — ExecuTorch Quantization, Delegates, and Edge Deployment](../originals/pytorch-systems-and-deployment-curriculum/modules/58-executorch-quantization-delegates-and-edge.md); [E18 — Week 18: ExecuTorch export, graph lowering and delegation](../originals/edge-inference-roadmap/weeks/week-18.md). Focus on delegate partition, using the task below to select the relevant subsection.

**External entry:** [ExecuTorch backend and quantization guides](https://docs.pytorch.org/executorch/stable/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect partitioned operators and CPU fallback boundaries.

**Done when:** Do not call partial delegation full-model accelerator execution.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-200.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-201"></a>

## Day 201 — Backend record

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [E20 — Week 20: Backend comparison and inference infrastructure](../originals/edge-inference-roadmap/weeks/week-20.md); [P57 — Module 57 — ExecuTorch Architecture, EXIR, and Backends](../originals/pytorch-systems-and-deployment-curriculum/modules/57-executorch-architecture-exir-and-backends.md). Focus on backend record, using the task below to select the relevant subsection.

**External entry:** [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md). See [resource verification notes](../RESOURCES.md).

**Do:** Add the host edge result to the common benchmark schema with artifact and backend identity.

**Done when:** Load/preparation costs remain separate from model execution.

**Career block:** Explain edge lowering aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-201.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-202"></a>

## Day 202 — Edge export gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P57 — Module 57 — ExecuTorch Architecture, EXIR, and Backends](../originals/pytorch-systems-and-deployment-curriculum/modules/57-executorch-architecture-exir-and-backends.md); [P58 — Module 58 — ExecuTorch Quantization, Delegates, and Edge Deployment](../originals/pytorch-systems-and-deployment-curriculum/modules/58-executorch-quantization-delegates-and-edge.md). Focus on edge export gate, using the task below to select the relevant subsection.

**External entry:** [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Package one reproducible host baseline plus a target-device compatibility checklist.

**Done when:** If export is blocked, retain a minimal reproducer and use the working native runtime route next.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use edge lowering as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-202.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain lowered graphs, unsupported operators and fallback coverage. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-203"></a>

## Day 203 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain lowered graphs, unsupported operators and fallback coverage.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
