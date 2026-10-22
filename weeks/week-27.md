# Week 27: DDP, sharding and failure recovery

Days 183–189 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Tiny distributed training check and reliability dossier.

**Prerequisite:** Pass the relevant gates from [week 26](week-26.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-183"></a>

## Day 183 — DDP baseline

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P44 — Module 44 — DistributedDataParallel](../originals/pytorch-systems-and-deployment-curriculum/modules/44-distributed-data-parallel.md). Focus on ddp baseline, using the task below to select the relevant subsection.

**External entry:** [DDP API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). See [resource verification notes](../RESOURCES.md).

**Do:** Run a tiny two-process CPU DDP example if available and compare one controlled update with a single-process reference.

**Done when:** Account for global batch size and reduction semantics.

**Career block:** Recall one coding or debugging pattern relevant to failure recovery; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-183.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-184"></a>

## Day 184 — FSDP survey

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P45 — Module 45 — FSDP and Sharded Training](../originals/pytorch-systems-and-deployment-curriculum/modules/45-fsdp-and-sharded-training.md). Focus on fsdp survey, using the task below to select the relevant subsection.

**External entry:** [FSDP API](https://docs.pytorch.org/docs/stable/fsdp.html). See [resource verification notes](../RESOURCES.md).

**Do:** Calculate parameter, gradient and optimizer-state memory for replicated and sharded cases.

**Done when:** Keep paper estimates separate from an actual FSDP run.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to failure recovery. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-184.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-185"></a>

## Day 185 — Parallelism decisions

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P47 — Module 47 — Tensor, Pipeline, Context, and Expert Parallelism](../originals/pytorch-systems-and-deployment-curriculum/modules/47-tensor-pipeline-context-and-expert-parallelism.md). Focus on parallelism decisions, using the task below to select the relevant subsection.

**External entry:** [PyTorch tensor parallel API](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html). See [resource verification notes](../RESOURCES.md).

**Do:** Compare tensor, pipeline, context and expert parallelism for two stated workloads.

**Done when:** Name communication and balance costs, not only memory savings.

**Career block:** Add one verified failure recovery observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-185.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-186"></a>

## Day 186 — Checkpoint recovery

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P48 — Module 48 — Distributed Checkpointing, Elasticity, and Reliability](../originals/pytorch-systems-and-deployment-curriculum/modules/48-distributed-checkpointing-elasticity-and-reliability.md); [P16 — Module 16 — Serialization, Checkpoints, and Model Artifacts](../originals/pytorch-systems-and-deployment-curriculum/modules/16-serialization-checkpoints-and-model-artifacts.md). Focus on checkpoint recovery, using the task below to select the relevant subsection.

**External entry:** [Distributed checkpointing](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html). See [resource verification notes](../RESOURCES.md).

**Do:** Interrupt a small local/distributed job at a safe checkpoint boundary and resume from committed state.

**Done when:** Reject incomplete artifacts and compare resumed outputs.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-186.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-187"></a>

## Day 187 — Topology inspection

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P49 — Module 49 — NCCL, Topology, Profiling, and Communication Performance](../originals/pytorch-systems-and-deployment-curriculum/modules/49-nccl-topology-profiling-and-communication-performance.md). Focus on topology inspection, using the task below to select the relevant subsection.

**External entry:** [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect NCCL documentation and an available trace or your CPU collective timings.

**Done when:** Label backend limitations; no fabricated NCCL/GPU result.

**Career block:** Explain failure recovery aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-187.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-188"></a>

## Day 188 — Distributed dossier

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P50 — Module 50 — Training–Serving Parallelism Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/50-training-serving-parallelism-bridge.md); [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md). Focus on distributed dossier, using the task below to select the relevant subsection.

**External entry:** [vLLM distributed serving](https://github.com/vllm-project/vllm). See [resource verification notes](../RESOURCES.md).

**Do:** Explain one inference deployment decision using workload, memory, topology and recovery needs.

**Done when:** Full multi-node training remains a later branch unless hardware and time are available.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use failure recovery as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-188.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain state ownership and recover a deliberately interrupted job. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-189"></a>

## Day 189 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain state ownership and recover a deliberately interrupted job.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
