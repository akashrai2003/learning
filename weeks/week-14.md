# Week 14: TorchAO, advanced quantization and low-bit limits

Days 092–098 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** One supported framework quantization workflow and comparison memo.

**Prerequisite:** Pass the relevant gates from [week 13](week-13.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-092"></a>

## Day 092 — TorchAO workflow

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P24 — Module 24 — TorchAO: Quantization, Sparsity, and Low-Bit Tensors](../originals/pytorch-systems-and-deployment-curriculum/modules/24-torchao-quantization-sparsity-and-low-bit.md); [I28 — Module 28 — Quantized Kernel and System Integration](../originals/inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md). Focus on torchao workflow, using the task below to select the relevant subsection.

**External entry:** [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Quantize a supported small layer/model using the version-matched TorchAO guide; inspect the resulting tensor/config.

**Done when:** Record the actual dispatch and any fallback rather than assuming acceleration.

**Career block:** Repeat one coding exercise under a shorter time limit. Use the [career reference](../originals/ai-career-acceleration/weeks/week-14.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-092.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-093"></a>

## Day 093 — GPTQ and AWQ comparison

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I22 — Module 22 — GPTQ and Approximate Second-Order PTQ](../originals/inference-engineering-curriculum/modules/22-gptq-second-order-ptq.md); [I23 — Module 23 — AWQ: Activation-Aware Weight Quantization](../originals/inference-engineering-curriculum/modules/23-awq-activation-aware-quantization.md). Focus on gptq and awq comparison, using the task below to select the relevant subsection.

**External entry:** [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). See [resource verification notes](../RESOURCES.md).

**Do:** Read the objective and core mechanism of each method; reproduce one tiny rescaling or error-compensation example.

**Done when:** Explain the different use of calibration information; full solvers are extensions.

**Career block:** Act on an interview/application next step. Use the [career reference](../originals/ai-career-acceleration/weeks/week-14.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-093.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-094"></a>

## Day 094 — Codebooks

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I24 — Module 24 — Low-Bit Formats: NF4, Codebooks, and GGUF](../originals/inference-engineering-curriculum/modules/24-low-bit-formats-nf4-codebooks-and-gguf.md). Focus on codebooks, using the task below to select the relevant subsection.

**External entry:** [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). See [resource verification notes](../RESOURCES.md).

**Do:** Compare uniform INT4 and a small nonuniform codebook on the same synthetic distribution.

**Done when:** Account for codebook and scale overhead and state the distribution assumption.

**Career block:** Draft a methods note about a controlled rerun. Use the [career reference](../originals/ai-career-acceleration/weeks/week-14.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-094.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-095"></a>

## Day 095 — FP8 and FP4

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I25 — Module 25 — FP8 Training and Inference](../originals/inference-engineering-curriculum/modules/25-fp8-training-and-inference.md); [I26 — Module 26 — Microscaling, MXFP4, and NVFP4](../originals/inference-engineering-curriculum/modules/26-microscaling-mxfp4-and-nvfp4.md). Focus on fp8 and fp4, using the task below to select the relevant subsection.

**External entry:** [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect format ranges and scaling; emulate one rounding example if native support is unavailable.

**Done when:** Label emulation clearly and make no native-throughput claim from it.

**Career block:** Collect missing terms only for any real offer or active process. Use the [career reference](../originals/ai-career-acceleration/weeks/week-14.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-095.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-096"></a>

## Day 096 — KV precision

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I27 — Module 27 — KV-Cache Quantization](../originals/inference-engineering-curriculum/modules/27-kv-cache-quantization.md); [I12 — Module 12 — MQA, GQA, and the KV Cache](../originals/inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md). Focus on kv precision, using the task below to select the relevant subsection.

**External entry:** [KIVI](https://arxiv.org/abs/2402.02750). See [resource verification notes](../RESOURCES.md).

**Do:** Quantize a small cache fixture and compare cached logits at two lengths.

**Done when:** Separate cache-memory savings from quality and conversion overhead.

**Career block:** Practice an ownership/incident follow-up question. Use the [career reference](../originals/ai-career-acceleration/weeks/week-14.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-096.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-097"></a>

## Day 097 — Quantization gate

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P24 — Module 24 — TorchAO: Quantization, Sparsity, and Low-Bit Tensors](../originals/pytorch-systems-and-deployment-curriculum/modules/24-torchao-quantization-sparsity-and-low-bit.md); [P28 — Module 28 — Evaluation, Quality, and Experiment Tracking](../originals/pytorch-systems-and-deployment-curriculum/modules/28-evaluation-quality-and-experiment-tracking.md). Focus on quantization gate, using the task below to select the relevant subsection.

**External entry:** [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html). See [resource verification notes](../RESOURCES.md).

**Do:** Select one supported quantization path for the flagship and preserve the float baseline.

**Done when:** Explain the chosen accuracy, size and latency tradeoff with evidence.

**Career block:** Polish one evidence artifact and run a mock; avoid starting a new project. Use the [career reference](../originals/ai-career-acceleration/weeks/week-14.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-097.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Defend a format choice from measured evidence. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-098"></a>

## Day 098 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Defend a format choice from measured evidence.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
