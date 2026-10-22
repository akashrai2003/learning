# Week 24: Buffer 2 — consolidate quantization, kernels and serving

Days 162–168 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Repaired optimization evidence and one selected specialization.

**Prerequisite:** Pass the relevant gates from [week 23](week-23.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-162"></a>

## Day 162 — Dependency audit

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on dependency audit, using the task below to select the relevant subsection.

**External entry:** [PyTorch testing overview](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#testing). See [resource verification notes](../RESOURCES.md).

**Do:** List unresolved gates from weeks 13–23; choose one correctness and one reproducibility repair.

**Done when:** Each repair is small enough to validate this week.

**Career block:** Recall one coding or debugging pattern relevant to six-month evidence audit; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-162.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-163"></a>

## Day 163 — Numerics repair

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I21 — Module 21 — Weight-Only INT8/INT4 and Packing](../originals/inference-engineering-curriculum/modules/21-weight-only-int8-int4-and-packing.md); [P24 — Module 24 — TorchAO: Quantization, Sparsity, and Low-Bit Tensors](../originals/pytorch-systems-and-deployment-curriculum/modules/24-torchao-quantization-sparsity-and-low-bit.md). Focus on numerics repair, using the task below to select the relevant subsection.

**External entry:** [TorchAO repository](https://github.com/pytorch/ao). See [resource verification notes](../RESOURCES.md).

**Do:** Re-run quantized packing, calibration and held-out checks on the smallest failing example.

**Done when:** No unexplained packed-format or quality regression remains.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to six-month evidence audit. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-163.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-164"></a>

## Day 164 — Compiler repair

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [P35 — Module 35 — `torch.library` and Python Custom Operators](../originals/pytorch-systems-and-deployment-curriculum/modules/35-torch-library-and-python-custom-ops.md); [P42 — Module 42 — Compiler Debugging, Minimization, and Regression Engineering](../originals/pytorch-systems-and-deployment-curriculum/modules/42-compiler-debugging-minimization-and-regression.md). Focus on compiler repair, using the task below to select the relevant subsection.

**External entry:** [torch.library](https://docs.pytorch.org/docs/stable/library.html). See [resource verification notes](../RESOURCES.md).

**Do:** Reconstruct the custom-op schema/fake implementation and re-run the dynamic input contract.

**Done when:** Unsupported cases fail clearly or fall back correctly.

**Career block:** Add one verified six-month evidence audit observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-164.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-165"></a>

## Day 165 — Kernel repair

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I36 — Module 36 — CUDA Execution, Memory, and Streams](../originals/inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md); [I29 — Module 29 — Attention IO Complexity and Online Softmax](../originals/inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md). Focus on kernel repair, using the task below to select the relevant subsection.

**External entry:** [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/). See [resource verification notes](../RESOURCES.md).

**Do:** Repeat the weakest tail, mask or synchronization test and inspect the first divergent output.

**Done when:** The repaired kernel still agrees with the untouched oracle.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-165.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-166"></a>

## Day 166 — Runtime repair

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I31 — Module 31 — PagedAttention and KV Memory Management](../originals/inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md); [I34 — Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache](../originals/inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md). Focus on runtime repair, using the task below to select the relevant subsection.

**External entry:** [Efficient Memory Management with PagedAttention](https://arxiv.org/abs/2309.06180). See [resource verification notes](../RESOURCES.md).

**Do:** Replay allocation, cancellation and long-request fairness cases in the simulator.

**Done when:** All requests have an explicit final state and no leaked blocks.

**Career block:** Explain six-month evidence audit aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-166.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-167"></a>

## Day 167 — Second-quarter review

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on second-quarter review, using the task below to select the relevant subsection.

**External entry:** [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html). See [resource verification notes](../RESOURCES.md).

**Do:** Choose the primary runtime and one deployment backend for the remaining plan.

**Done when:** Document deferred alternatives and keep one upstream work item active at most.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use six-month evidence audit as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-167.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Close correctness gaps before adding distributed or mobile environments. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-168"></a>

## Day 168 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Close correctness gaps before adding distributed or mobile environments.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
