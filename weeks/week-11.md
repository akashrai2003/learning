# Week 11: INT8 quantization and calibration

Days 071–077 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Scalar quantizer and calibration/evaluation matrix.

**Prerequisite:** Pass the relevant gates from [week 10](week-10.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-071"></a>

## Day 071 — Affine quantization

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I18 — Module 18 — Quantization Fundamentals](../originals/inference-engineering-curriculum/modules/18-quantization-fundamentals.md). Focus on affine quantization, using the task below to select the relevant subsection.

**External entry:** [Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877). See [resource verification notes](../RESOURCES.md).

**Do:** Derive scale and zero point; implement scalar quantize/dequantize with a defined constant-range policy.

**Done when:** Test zero, constants, clipping and signed-range conventions.

**Career block:** Practice one weak coding pattern with a new example. Use the [career reference](../originals/ai-career-acceleration/weeks/week-11.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-071.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-072"></a>

## Day 072 — Calibration

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I19 — Module 19 — Calibration, Error Metrics, and Evaluation](../originals/inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md); [I20 — Module 20 — INT8: Static, Dynamic, and SmoothQuant](../originals/inference-engineering-curriculum/modules/20-int8-static-dynamic-and-smoothquant.md). Focus on calibration, using the task below to select the relevant subsection.

**External entry:** [ACIQ: Analytical Clipping for Integer Quantization](https://arxiv.org/abs/1810.05723). See [resource verification notes](../RESOURCES.md).

**Do:** Compare min/max calibration with a simple clipping choice on representative inputs.

**Done when:** Keep calibration and held-out evaluation samples separate.

**Career block:** Apply or follow up manually using current eligibility information. Use the [career reference](../originals/ai-career-acceleration/weeks/week-11.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-072.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-073"></a>

## Day 073 — Granularity

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I18 — Module 18 — Quantization Fundamentals](../originals/inference-engineering-curriculum/modules/18-quantization-fundamentals.md); [P24 — Module 24 — TorchAO: Quantization, Sparsity, and Low-Bit Tensors](../originals/pytorch-systems-and-deployment-curriculum/modules/24-torchao-quantization-sparsity-and-low-bit.md). Focus on granularity, using the task below to select the relevant subsection.

**External entry:** [Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877). See [resource verification notes](../RESOURCES.md).

**Do:** Implement per-tensor and per-channel quantization for one linear layer.

**Done when:** Broadcast axes and reconstructed weights match the written format contract.

**Career block:** Write a source-level explanation linked to an actual code path. Use the [career reference](../originals/ai-career-acceleration/weeks/week-11.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-073.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-074"></a>

## Day 074 — Activation outliers

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I20 — Module 20 — INT8: Static, Dynamic, and SmoothQuant](../originals/inference-engineering-curriculum/modules/20-int8-static-dynamic-and-smoothquant.md). Focus on activation outliers, using the task below to select the relevant subsection.

**External entry:** [SmoothQuant: Accurate and Efficient PTQ for LLMs](https://arxiv.org/abs/2211.10438). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect outlier channels and derive the rescaling identity used by SmoothQuant on a tiny layer.

**Done when:** The full-precision rescaling preserves outputs before quantization.

**Career block:** Maintain the contribution reproduction and its pinned revision. Use the [career reference](../originals/ai-career-acceleration/weeks/week-11.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-074.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-075"></a>

## Day 075 — Quality versus size

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I19 — Module 19 — Calibration, Error Metrics, and Evaluation](../originals/inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md); [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md). Focus on quality versus size, using the task below to select the relevant subsection.

**External entry:** [ACIQ: Analytical Clipping for Integer Quantization](https://arxiv.org/abs/1810.05723). See [resource verification notes](../RESOURCES.md).

**Do:** Measure layer error, serialized bytes and held-out quality for the INT8 experiment.

**Done when:** Do not infer speed or model quality from weight MSE alone.

**Career block:** Rehearse a performance interview using the latency/quality baseline. Use the [career reference](../originals/ai-career-acceleration/weeks/week-11.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-075.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-076"></a>

## Day 076 — INT8 gate

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [E12 — Week 12: INT8 quantization and quality evaluation](../originals/edge-inference-roadmap/weeks/week-12.md); [I20 — Module 20 — INT8: Static, Dynamic, and SmoothQuant](../originals/inference-engineering-curriculum/modules/20-int8-static-dynamic-and-smoothquant.md). Focus on int8 gate, using the task below to select the relevant subsection.

**External entry:** [ggml source and quantization implementation](https://github.com/ggml-org/llama.cpp/tree/master/ggml). See [resource verification notes](../RESOURCES.md).

**Do:** Record a reproducible comparison and reconstruct qparams by hand for one vector.

**Done when:** Explain every scale, zero point and accumulation dtype.

**Career block:** Run a mock and refine the strongest case study for a real target role. Use the [career reference](../originals/ai-career-acceleration/weeks/week-11.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-076.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain qparams and separate calibration data from evaluation data. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-077"></a>

## Day 077 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain qparams and separate calibration data from evaluation data.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
