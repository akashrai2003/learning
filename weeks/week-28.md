# Week 28: torch.export, ONNX and artifact contracts

Days 190–196 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Exported small model and runtime parity report.

**Prerequisite:** Pass the relevant gates from [week 27](week-27.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-190"></a>

## Day 190 — Export contract

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P51 — Module 51 — `torch.export` and Program Capture](../originals/pytorch-systems-and-deployment-curriculum/modules/51-torch-export-and-program-capture.md). Focus on export contract, using the task below to select the relevant subsection.

**External entry:** [torch.export](https://docs.pytorch.org/docs/stable/export.html). See [resource verification notes](../RESOURCES.md).

**Do:** Export a supported small module with explicit input dimensions and state.

**Done when:** Exercise the lower/upper allowed bounds and reject an out-of-contract input.

**Career block:** Recall one coding or debugging pattern relevant to export contracts; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-190.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-191"></a>

## Day 191 — Export debugging

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P51 — Module 51 — `torch.export` and Program Capture](../originals/pytorch-systems-and-deployment-curriculum/modules/51-torch-export-and-program-capture.md); [P35 — Module 35 — `torch.library` and Python Custom Operators](../originals/pytorch-systems-and-deployment-curriculum/modules/35-torch-library-and-python-custom-ops.md). Focus on export debugging, using the task below to select the relevant subsection.

**External entry:** [torch.export](https://docs.pytorch.org/docs/stable/export.html). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect exported operators and isolate one unsupported behavior in a small example.

**Done when:** Document the precise constraint or legal rewrite.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to export contracts. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-191.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-192"></a>

## Day 192 — ONNX artifact

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P52 — Module 52 — ONNX Export and ONNX IR](../originals/pytorch-systems-and-deployment-curriculum/modules/52-onnx-export-and-onnx-ir.md). Focus on onnx artifact, using the task below to select the relevant subsection.

**External entry:** [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html). See [resource verification notes](../RESOURCES.md).

**Do:** Export the same supported module to ONNX and inspect inputs, outputs and opset metadata.

**Done when:** The artifact describes the intended dtype and dynamic dimensions.

**Career block:** Add one verified export contracts observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-192.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-193"></a>

## Day 193 — ONNX Runtime parity

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P53 — Module 53 — ONNX Runtime and Execution Providers](../originals/pytorch-systems-and-deployment-curriculum/modules/53-onnx-runtime-and-execution-providers.md). Focus on onnx runtime parity, using the task below to select the relevant subsection.

**External entry:** [ONNX Runtime docs](https://onnxruntime.ai/docs/). See [resource verification notes](../RESOURCES.md).

**Do:** Run the model in an available ONNX Runtime execution provider.

**Done when:** Compare outputs with eager PyTorch across the declared input contract.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-193.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-194"></a>

## Day 194 — Packaging survey

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P56 — Module 56 — AOTInductor Packaging and Native Deployment](../originals/pytorch-systems-and-deployment-curriculum/modules/56-aotinductor-packaging-and-native-deployment.md). Focus on packaging survey, using the task below to select the relevant subsection.

**External entry:** [AOTInductor documentation](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect AOTInductor packaging and compare its deployment dependencies with ONNX Runtime.

**Done when:** Write an artifact decision table; build a second package only if the core export is complete.

**Career block:** Explain export contracts aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-194.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-195"></a>

## Day 195 — Deployment gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P51 — Module 51 — `torch.export` and Program Capture](../originals/pytorch-systems-and-deployment-curriculum/modules/51-torch-export-and-program-capture.md); [P53 — Module 53 — ONNX Runtime and Execution Providers](../originals/pytorch-systems-and-deployment-curriculum/modules/53-onnx-runtime-and-execution-providers.md). Focus on deployment gate, using the task below to select the relevant subsection.

**External entry:** [torch.export](https://docs.pytorch.org/docs/stable/export.html). See [resource verification notes](../RESOURCES.md).

**Do:** Recreate one exported artifact and its parity report from recorded commands.

**Done when:** Separate export time, runtime load time and steady execution.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use export contracts as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-195.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Declare input constraints and verify outputs in a second runtime. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-196"></a>

## Day 196 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Declare input constraints and verify outputs in a second runtime.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
