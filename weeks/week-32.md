# Week 32: Architecture breadth and one focused model port

Days 218–224 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Architecture dossier and one verified adaptation.

**Prerequisite:** Pass the relevant gates from [week 31](week-31.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-218"></a>

## Day 218 — MoE routing

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I13 — Module 13 — Mixture of Experts](../originals/inference-engineering-curriculum/modules/13-mixture-of-experts.md). Focus on moe routing, using the task below to select the relevant subsection.

**External entry:** [“Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”](https://arxiv.org/abs/1701.06538). See [resource verification notes](../RESOURCES.md).

**Do:** Implement a tiny top-k expert router and inspect capacity/load distribution.

**Done when:** Routing weights and expert output recombination match a hand example.

**Career block:** Recall one coding or debugging pattern relevant to architecture boundaries; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-218.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-219"></a>

## Day 219 — Multimodal contracts

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I15 — Module 15 — Multimodal Inference](../originals/inference-engineering-curriculum/modules/15-multimodal-inference.md); [P27 — Module 27 — Multimodal Models and Processors](../originals/pytorch-systems-and-deployment-curriculum/modules/27-multimodal-models-and-processors.md). Focus on multimodal contracts, using the task below to select the relevant subsection.

**External entry:** [“Learning Transferable Visual Models From Natural Language Supervision”](https://arxiv.org/abs/2103.00020). See [resource verification notes](../RESOURCES.md).

**Do:** Trace one supported model's processor, placeholder tokens and embedding boundary.

**Done when:** Record exact shapes; a source walk is not a completed multimodal implementation.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to architecture boundaries. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-219.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-220"></a>

## Day 220 — Recurrent and hybrid state

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I16 — Module 16 — SSM, Recurrent, and Hybrid Models](../originals/inference-engineering-curriculum/modules/16-ssm-recurrent-and-hybrid-models.md). Focus on recurrent and hybrid state, using the task below to select the relevant subsection.

**External entry:** [“Mamba: Linear-Time Sequence Modeling with Selective State Spaces”](https://arxiv.org/abs/2312.00752). See [resource verification notes](../RESOURCES.md).

**Do:** Compare attention KV state with one SSM/recurrent model's documented state interface.

**Done when:** Write a reset/update contract and a small state-size calculation.

**Career block:** Add one verified architecture boundaries observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-220.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-221"></a>

## Day 221 — Vision/audio survey

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P25 — Module 25 — Torchvision and Vision Model Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/25-torchvision-and-vision-model-engineering.md); [P26 — Module 26 — TorchAudio, TorchCodec, and Streaming Audio](../originals/pytorch-systems-and-deployment-curriculum/modules/26-torchaudio-and-streaming-audio.md). Focus on vision/audio survey, using the task below to select the relevant subsection.

**External entry:** [Torchvision docs](https://docs.pytorch.org/vision/stable/). See [resource verification notes](../RESOURCES.md).

**Do:** Compare a vision preprocessing path and streaming-audio boundary from canonical examples.

**Done when:** List batching, normalization and streaming-state requirements; full apps are deferred.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-221.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-222"></a>

## Day 222 — Focused adaptation

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I17 — Module 17 — Modern Model Architecture Forensics](../originals/inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md); [P27 — Module 27 — Multimodal Models and Processors](../originals/pytorch-systems-and-deployment-curriculum/modules/27-multimodal-models-and-processors.md). Focus on focused adaptation, using the task below to select the relevant subsection.

**External entry:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762). See [resource verification notes](../RESOURCES.md).

**Do:** Adapt one compatible configuration or input boundary in the reference model; choose a small verified change.

**Done when:** Intermediate and final outputs agree with the chosen reference for the supported subset.

**Career block:** Explain architecture boundaries aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-222.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-223"></a>

## Day 223 — Model-port dossier

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [I17 — Module 17 — Modern Model Architecture Forensics](../originals/inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md); [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md). Focus on model-port dossier, using the task below to select the relevant subsection.

**External entry:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762). See [resource verification notes](../RESOURCES.md).

**Do:** Document configuration, checkpoint mapping, unsupported ops, quality and resource impact.

**Done when:** Do not describe a config inspection alone as a full production model port.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use architecture boundaries as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-223.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain how model differences affect state, shapes and runtime support. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-224"></a>

## Day 224 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain how model differences affect state, shapes and runtime support.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
