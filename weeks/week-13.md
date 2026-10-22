# Week 13: INT4 packing and real storage cost

Days 085–091 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Bit-exact packed format and decode/prefill comparison.

**Prerequisite:** Pass the relevant gates from [week 12](week-12.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-085"></a>

## Day 085 — Packing contract

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I21 — Module 21 — Weight-Only INT8/INT4 and Packing](../originals/inference-engineering-curriculum/modules/21-weight-only-int8-int4-and-packing.md). Focus on packing contract, using the task below to select the relevant subsection.

**External entry:** [TorchAO repository](https://github.com/pytorch/ao). See [resource verification notes](../RESOURCES.md).

**Do:** Specify signed mapping, nibble order, group size, padding and scales; implement pack/unpack.

**Done when:** All representable codes round-trip exactly, including odd lengths.

**Career block:** Revisit a practical coding weakness with tests. Use the [career reference](../originals/ai-career-acceleration/weeks/week-13.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-085.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-086"></a>

## Day 086 — Grouped quantization

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I18 — Module 18 — Quantization Fundamentals](../originals/inference-engineering-curriculum/modules/18-quantization-fundamentals.md); [I21 — Module 21 — Weight-Only INT8/INT4 and Packing](../originals/inference-engineering-curriculum/modules/21-weight-only-int8-int4-and-packing.md). Focus on grouped quantization, using the task below to select the relevant subsection.

**External entry:** [Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877). See [resource verification notes](../RESOURCES.md).

**Do:** Quantize one linear layer in groups and serialize a versioned format manifest.

**Done when:** Non-divisible groups and constant groups reconstruct correctly.

**Career block:** Take a high-fit application step using quantization evidence. Use the [career reference](../originals/ai-career-acceleration/weeks/week-13.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-086.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-087"></a>

## Day 087 — Format inspection

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E13 — Week 13: INT4 packing, group quantization and fused dequantization](../originals/edge-inference-roadmap/weeks/week-13.md); [I24 — Module 24 — Low-Bit Formats: NF4, Codebooks, and GGUF](../originals/inference-engineering-curriculum/modules/24-low-bit-formats-nf4-codebooks-and-gguf.md). Focus on format inspection, using the task below to select the relevant subsection.

**External entry:** [ggml source and quantization implementation](https://github.com/ggml-org/llama.cpp/tree/master/ggml). See [resource verification notes](../RESOURCES.md).

**Do:** Compare your educational format with one GGUF tensor format in canonical source.

**Done when:** Explain why your INT4 file is not automatically runtime-compatible GGUF.

**Career block:** Explain a verified format/metadata lesson; avoid claiming phone results not yet measured. Use the [career reference](../originals/ai-career-acceleration/weeks/week-13.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-087.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-088"></a>

## Day 088 — Linear integration

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I21 — Module 21 — Weight-Only INT8/INT4 and Packing](../originals/inference-engineering-curriculum/modules/21-weight-only-int8-int4-and-packing.md); [I28 — Module 28 — Quantized Kernel and System Integration](../originals/inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md). Focus on linear integration, using the task below to select the relevant subsection.

**External entry:** [TorchAO repository](https://github.com/pytorch/ao). See [resource verification notes](../RESOURCES.md).

**Do:** Add a visible quantized-linear path with a safe reference fallback.

**Done when:** The dispatch rejects unsupported layout combinations explicitly.

**Career block:** Maintain or defer the existing contribution item explicitly. Use the [career reference](../originals/ai-career-acceleration/weeks/week-13.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-088.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-089"></a>

## Day 089 — Storage and speed

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I05 — Module 05 — Performance Modeling and Roofline](../originals/inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md); [I21 — Module 21 — Weight-Only INT8/INT4 and Packing](../originals/inference-engineering-curriculum/modules/21-weight-only-int8-int4-and-packing.md). Focus on storage and speed, using the task below to select the relevant subsection.

**External entry:** [Roofline: An Insightful Visual Performance Model](https://digicoll.lib.berkeley.edu/record/136692/files/EECS-2008-134.pdf). See [resource verification notes](../RESOURCES.md).

**Do:** Measure packed bytes including metadata and dequantization cost for small and large M dimensions.

**Done when:** Include conversion cost and a case where compression does not improve latency.

**Career block:** Rehearse an architecture tradeoff using latency, quality and memory. Use the [career reference](../originals/ai-career-acceleration/weeks/week-13.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-089.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-090"></a>

## Day 090 — Quantization report

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [I19 — Module 19 — Calibration, Error Metrics, and Evaluation](../originals/inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md); [I21 — Module 21 — Weight-Only INT8/INT4 and Packing](../originals/inference-engineering-curriculum/modules/21-weight-only-int8-int4-and-packing.md). Focus on quantization report, using the task below to select the relevant subsection.

**External entry:** [ACIQ: Analytical Clipping for Integer Quantization](https://arxiv.org/abs/1810.05723). See [resource verification notes](../RESOURCES.md).

**Do:** Combine INT8 and INT4 results into one quality/bytes/latency table.

**Done when:** Every row uses the same workload and a documented tolerance.

**Career block:** Run a mock and have a relevant professional conversation only if there is a concrete purpose. Use the [career reference](../originals/ai-career-acceleration/weeks/week-13.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-090.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Handle group tails and account for metadata bytes. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-091"></a>

## Day 091 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Handle group tails and account for metadata bytes.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
