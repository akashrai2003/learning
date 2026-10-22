# Week 33: Runtime alternatives and deployment choices

Days 225–231 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Evidence-based deployment decision record.

**Prerequisite:** Pass the relevant gates from [week 32](week-32.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-225"></a>

## Day 225 — SGLang source survey

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I47 — Module 47 — SGLang Internals and Contributions](../originals/inference-engineering-curriculum/modules/47-sglang-internals-and-contributions.md). Focus on sglang source survey, using the task below to select the relevant subsection.

**External entry:** [SGLang source](https://github.com/sgl-project/sglang). See [resource verification notes](../RESOURCES.md).

**Do:** Trace one request/cache path in a pinned SGLang checkout and compare with the primary runtime.

**Done when:** Identify a concrete policy difference; running a second full server is optional.

**Career block:** Recall one coding or debugging pattern relevant to deployment decisions; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-225.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-226"></a>

## Day 226 — llama.cpp/ggml survey

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I48 — Module 48 — llama.cpp/ggml Internals and Contributions](../originals/inference-engineering-curriculum/modules/48-llamacpp-ggml-internals-and-contributions.md); [E20 — Week 20: Backend comparison and inference infrastructure](../originals/edge-inference-roadmap/weeks/week-20.md). Focus on llama.cpp/ggml survey, using the task below to select the relevant subsection.

**External entry:** [llama.cpp source](https://github.com/ggml-org/llama.cpp). See [resource verification notes](../RESOURCES.md).

**Do:** Trace one quantized tensor from loader to kernel dispatch using the established source skills.

**Done when:** Separate storage format, compute path and hardware backend.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to deployment decisions. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-226.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-227"></a>

## Day 227 — TensorRT option

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P54 — Module 54 — TensorRT and Torch-TensorRT](../originals/pytorch-systems-and-deployment-curriculum/modules/54-tensorrt-and-torch-tensorrt.md). Focus on tensorrt option, using the task below to select the relevant subsection.

**External entry:** [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect supported input contracts and target requirements; run a tiny example only if the setup is already feasible.

**Done when:** State engine build constraints and parity checks needed for the target.

**Career block:** Add one verified deployment decisions observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-227.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-228"></a>

## Day 228 — TensorRT-LLM option

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P55 — Module 55 — TensorRT-LLM and Runtime Integration](../originals/pytorch-systems-and-deployment-curriculum/modules/55-tensorrt-llm-and-runtime-integration.md). Focus on tensorrt-llm option, using the task below to select the relevant subsection.

**External entry:** [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/). See [resource verification notes](../RESOURCES.md).

**Do:** Compare artifact, model-support and runtime assumptions with the primary server.

**Done when:** Produce a source-based decision note rather than an untested speed ranking.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-228.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-229"></a>

## Day 229 — Deployment matrix

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P56 — Module 56 — AOTInductor Packaging and Native Deployment](../originals/pytorch-systems-and-deployment-curriculum/modules/56-aotinductor-packaging-and-native-deployment.md); [P06 — Module 06 — Framework Landscape and Interoperability](../originals/pytorch-systems-and-deployment-curriculum/modules/06-framework-landscape-and-interoperability.md). Focus on deployment matrix, using the task below to select the relevant subsection.

**External entry:** [AOTInductor documentation](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html). See [resource verification notes](../RESOURCES.md).

**Do:** Compare eager/compiled PyTorch, ONNX Runtime, ExecuTorch and the chosen LLM runtime for your workload.

**Done when:** Support every measured cell with a result; label all others untested or unsupported.

**Career block:** Explain deployment decisions aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-229.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-230"></a>

## Day 230 — Capstone design freeze

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P62 — Module 62 — PyTorch Ecosystem Capstone](../originals/pytorch-systems-and-deployment-curriculum/modules/62-pytorch-ecosystem-capstone.md). Focus on capstone design freeze, using the task below to select the relevant subsection.

**External entry:** [PyTorch ecosystem](https://pytorch.org/ecosystem/). See [resource verification notes](../RESOURCES.md).

**Do:** Choose one primary serving or edge capstone path and one small contrast experiment.

**Done when:** Freeze model, data, quality goals and runtime scope for the final six weeks.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use deployment decisions as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-230.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Choose one deployment route and explain rejected alternatives. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-231"></a>

## Day 231 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Choose one deployment route and explain rejected alternatives.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
