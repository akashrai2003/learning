# Week 12: Buffer 1 — repair fundamentals and assess the first quarter

Days 078–084 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Repaired prerequisite gates and a realistic next-quarter scope.

**Prerequisite:** Pass the relevant gates from [week 11](week-11.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-078"></a>

## Day 078 — Triage

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on triage, using the task below to select the relevant subsection.

**External entry:** [PyTorch testing overview](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#testing). See [resource verification notes](../RESOURCES.md).

**Do:** List failed gates from weeks 1–11 and rank them by dependency impact; select at most two repairs.

**Done when:** Each selected blocker has a failing input and a clear acceptance check.

**Career block:** Audit two career stories for unsupported claims. Use the [career reference](../originals/ai-career-acceleration/weeks/week-12.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-078.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-079"></a>

## Day 079 — Foundation repair

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P07 — Module 07 — Tensor Storage, Views, Strides, and Layouts](../originals/pytorch-systems-and-deployment-curriculum/modules/07-tensor-storage-views-strides-and-layouts.md); [P10 — Module 10 — Autograd Engine and Gradient Modes](../originals/pytorch-systems-and-deployment-curriculum/modules/10-autograd-engine-and-gradient-modes.md). Focus on foundation repair, using the task below to select the relevant subsection.

**External entry:** [Tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html). See [resource verification notes](../RESOURCES.md).

**Do:** Reconstruct the weakest tensor, ownership or gradient exercise without notes; repair its failure.

**Done when:** Use a fresh example rather than replaying memorized code.

**Career block:** Act on the best active interview or application next step. Use the [career reference](../originals/ai-career-acceleration/weeks/week-12.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-079.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-080"></a>

## Day 080 — Decoder repair

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I12 — Module 12 — MQA, GQA, and the KV Cache](../originals/inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md); [I17 — Module 17 — Modern Model Architecture Forensics](../originals/inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md). Focus on decoder repair, using the task below to select the relevant subsection.

**External entry:** [“Fast Transformer Decoding: One Write-Head is All You Need”](https://arxiv.org/abs/1911.02150). See [resource verification notes](../RESOURCES.md).

**Do:** Re-run checkpoint and cached-generation parity on the smallest failing example.

**Done when:** Find the first wrong intermediate or document an explicitly unsupported architecture.

**Career block:** Condense the first-quarter evidence into a short resume update. Use the [career reference](../originals/ai-career-acceleration/weeks/week-12.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-080.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-081"></a>

## Day 081 — Quantization repair

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I18 — Module 18 — Quantization Fundamentals](../originals/inference-engineering-curriculum/modules/18-quantization-fundamentals.md); [I19 — Module 19 — Calibration, Error Metrics, and Evaluation](../originals/inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md). Focus on quantization repair, using the task below to select the relevant subsection.

**External entry:** [Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877). See [resource verification notes](../RESOURCES.md).

**Do:** Recheck constant scales, axes, calibration leakage and evaluation definitions.

**Done when:** A corrected quantizer still passes the unmodified baseline tests.

**Career block:** List offer questions and decision criteria without assuming an offer exists. Use the [career reference](../originals/ai-career-acceleration/weeks/week-12.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-081.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-082"></a>

## Day 082 — Evidence repair

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P18 — Module 18 — PyTorch Profiler, Memory, and Diagnostics](../originals/pytorch-systems-and-deployment-curriculum/modules/18-pytorch-profiler-memory-and-diagnostics.md); [I44 — Module 44 — Profiling, Benchmarking, and Correctness](../originals/inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). Focus on evidence repair, using the task below to select the relevant subsection.

**External entry:** [Profiler API](https://docs.pytorch.org/docs/stable/profiler.html). See [resource verification notes](../RESOURCES.md).

**Do:** Regenerate one report in a clean output directory and remove unsupported performance claims.

**Done when:** Raw records, scripts and narrative agree.

**Career block:** Practice a design follow-up under a changed memory constraint. Use the [career reference](../originals/ai-career-acceleration/weeks/week-12.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-082.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-083"></a>

## Day 083 — Quarter review

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P05 — Module 05 — Testing, Debugging, and Benchmark Discipline](../originals/pytorch-systems-and-deployment-curriculum/modules/05-testing-debugging-and-benchmark-discipline.md); [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md). Focus on quarter review, using the task below to select the relevant subsection.

**External entry:** [PyTorch testing overview](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#testing). See [resource verification notes](../RESOURCES.md).

**Do:** Use MILESTONES.md to mark pass, repeat or deferred for each gate; plan the next six weeks from evidence.

**Done when:** If a prerequisite remains broken, shift dependent weeks; never silently mark it complete.

**Career block:** Run a mock, review the three-month pipeline and select one concrete correction. Use the [career reference](../originals/ai-career-acceleration/weeks/week-12.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-083.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Close the highest-impact correctness gaps before new precision work. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-084"></a>

## Day 084 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Close the highest-impact correctness gaps before new precision work.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
