# Week 26: Distributed fundamentals on available hardware

Days 176–182 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Two-process collective lab and inference communication model.

**Prerequisite:** Pass the relevant gates from [week 25](week-25.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-176"></a>

## Day 176 — Collectives

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P43 — Module 43 — Process Groups, Collectives, and `torch.distributed`](../originals/pytorch-systems-and-deployment-curriculum/modules/43-process-groups-collectives-and-torch-distributed.md); [I45 — Module 45 — Distributed Inference Parallelism](../originals/inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md). Focus on collectives, using the task below to select the relevant subsection.

**External entry:** [torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html). See [resource verification notes](../RESOURCES.md).

**Do:** Draw all-reduce, all-gather and reduce-scatter on small arrays.

**Done when:** Compute expected outputs by hand before running code.

**Career block:** Recall one coding or debugging pattern relevant to distributed semantics; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-176.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-177"></a>

## Day 177 — Two-process lab

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P43 — Module 43 — Process Groups, Collectives, and `torch.distributed`](../originals/pytorch-systems-and-deployment-curriculum/modules/43-process-groups-collectives-and-torch-distributed.md). Focus on two-process lab, using the task below to select the relevant subsection.

**External entry:** [torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html). See [resource verification notes](../RESOURCES.md).

**Do:** Run a tiny CPU/Gloo collective example if supported; use explicit rank/world-size setup and timeouts.

**Done when:** Each rank gets the expected result and exits cleanly.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to distributed semantics. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-177.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-178"></a>

## Day 178 — Serving parallelism

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I45 — Module 45 — Distributed Inference Parallelism](../originals/inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md); [P50 — Module 50 — Training–Serving Parallelism Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/50-training-serving-parallelism-bridge.md). Focus on serving parallelism, using the task below to select the relevant subsection.

**External entry:** [Megatron-LM paper](https://arxiv.org/abs/1909.08053). See [resource verification notes](../RESOURCES.md).

**Do:** Map tensor and pipeline parallelism onto the reference decoder's tensors and cache.

**Done when:** State which activations move and which weights/cache remain local.

**Career block:** Add one verified distributed semantics observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-178.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-179"></a>

## Day 179 — Placement model

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P46 — Module 46 — DeviceMesh and DTensor](../originals/pytorch-systems-and-deployment-curriculum/modules/46-devicemesh-and-dtensor.md). Focus on placement model, using the task below to select the relevant subsection.

**External entry:** [DeviceMesh API](https://docs.pytorch.org/docs/stable/distributed.tensor.html). See [resource verification notes](../RESOURCES.md).

**Do:** Work through a small DeviceMesh/DTensor placement example using supported hardware or a labeled simulation.

**Done when:** Explain shard versus replicate and the required redistribution.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-179.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-180"></a>

## Day 180 — Communication cost

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P49 — Module 49 — NCCL, Topology, Profiling, and Communication Performance](../originals/pytorch-systems-and-deployment-curriculum/modules/49-nccl-topology-profiling-and-communication-performance.md); [I45 — Module 45 — Distributed Inference Parallelism](../originals/inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md). Focus on communication cost, using the task below to select the relevant subsection.

**External entry:** [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/). See [resource verification notes](../RESOURCES.md).

**Do:** Estimate bytes and latency terms for one collective and compare only with measurements you can actually make.

**Done when:** Do not infer multi-GPU scaling from CPU processes or one GPU.

**Career block:** Explain distributed semantics aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-180.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-181"></a>

## Day 181 — Distributed gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P43 — Module 43 — Process Groups, Collectives, and `torch.distributed`](../originals/pytorch-systems-and-deployment-curriculum/modules/43-process-groups-collectives-and-torch-distributed.md); [P50 — Module 50 — Training–Serving Parallelism Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/50-training-serving-parallelism-bridge.md). Focus on distributed gate, using the task below to select the relevant subsection.

**External entry:** [torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html). See [resource verification notes](../RESOURCES.md).

**Do:** Write a training-versus-serving parallelism comparison with one failure scenario.

**Done when:** The report names topology, backend and the limits of the local experiment.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use distributed semantics as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-181.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Distinguish semantic experiments from multi-GPU performance evidence. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-182"></a>

## Day 182 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Distinguish semantic experiments from multi-GPU performance evidence.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
