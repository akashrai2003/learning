# Week 08: Positions, GQA and cached decoding

Days 050–056 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Token-by-token cached decoder and cache budget.

**Prerequisite:** Pass the relevant gates from [week 07](week-07.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-050"></a>

## Day 050 — Rotary positions

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I11 — Module 11 — Positional Encoding, RoPE, and Scaling](../originals/inference-engineering-curriculum/modules/11-positional-encoding-rope-and-scaling.md). Focus on rotary positions, using the task below to select the relevant subsection.

**External entry:** [“RoFormer: Enhanced Transformer with Rotary Position Embedding”](https://arxiv.org/abs/2104.09864). See [resource verification notes](../RESOURCES.md).

**Do:** Implement RoPE with explicit position indices and compare a small hand-calculated rotation.

**Done when:** Cache offsets and reset positions are covered by tests.

**Career block:** Repeat one early coding problem without notes and compare quality. Use the [career reference](../originals/ai-career-acceleration/weeks/week-08.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-050.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-051"></a>

## Day 051 — GQA shapes

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I12 — Module 12 — MQA, GQA, and the KV Cache](../originals/inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md); [P21 — Module 21 — Attention, KV Cache, and the PyTorch Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/21-attention-kv-cache-and-pytorch-bridge.md). Focus on gqa shapes, using the task below to select the relevant subsection.

**External entry:** [“Fast Transformer Decoding: One Write-Head is All You Need”](https://arxiv.org/abs/1911.02150). See [resource verification notes](../RESOURCES.md).

**Do:** Implement grouped-query attention with a small number of KV heads; calculate cache bytes.

**Done when:** Head mapping is correct and non-divisible head counts are rejected.

**Career block:** Act on a role using the stronger cached-decoder evidence. Use the [career reference](../originals/ai-career-acceleration/weeks/week-08.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-051.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-052"></a>

## Day 052 — KV-cache update

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I12 — Module 12 — MQA, GQA, and the KV Cache](../originals/inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md). Focus on kv-cache update, using the task below to select the relevant subsection.

**External entry:** [“Fast Transformer Decoding: One Write-Head is All You Need”](https://arxiv.org/abs/1911.02150). See [resource verification notes](../RESOURCES.md).

**Do:** Add an append-only cache and a reset path; compare full and incremental logits.

**Done when:** Test one-token, multi-token prefill and cache-capacity boundaries.

**Career block:** Draft a concise evidence summary for the profile or resume. Use the [career reference](../originals/ai-career-acceleration/weeks/week-08.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-052.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-053"></a>

## Day 053 — Sampling and stopping

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I32 — Module 32 — Decoding, Sampling, and Logits Processing](../originals/inference-engineering-curriculum/modules/32-decoding-sampling-and-logits-processing.md). Focus on sampling and stopping, using the task below to select the relevant subsection.

**External entry:** [The Curious Case of Neural Text Degeneration / nucleus sampling](https://arxiv.org/abs/1904.09751). See [resource verification notes](../RESOURCES.md).

**Do:** Implement greedy selection, temperature and top-k with clear stop-token and maximum-length rules.

**Done when:** Extreme logits remain numerically stable and stop behavior is deterministic in greedy mode.

**Career block:** Maintain the one contribution item or explicitly defer it. Use the [career reference](../originals/ai-career-acceleration/weeks/week-08.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-053.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-054"></a>

## Day 054 — Long-context costs

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I14 — Module 14 — Long Context and Memory](../originals/inference-engineering-curriculum/modules/14-long-context-and-memory.md); [I05 — Module 05 — Performance Modeling and Roofline](../originals/inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md). Focus on long-context costs, using the task below to select the relevant subsection.

**External entry:** [“Efficient Memory Management for Large Language Model Serving with PagedAttention”](https://arxiv.org/abs/2309.06180). See [resource verification notes](../RESOURCES.md).

**Do:** Estimate cache memory at three lengths and measure the feasible small cases.

**Done when:** Separate estimates from measurements and document the first resource constraint.

**Career block:** Practice a short written technical handoff for remote collaboration. Use the [career reference](../originals/ai-career-acceleration/weeks/week-08.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-054.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-055"></a>

## Day 055 — Cached-decoder gate

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [P21 — Module 21 — Attention, KV Cache, and the PyTorch Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/21-attention-kv-cache-and-pytorch-bridge.md); [I12 — Module 12 — MQA, GQA, and the KV Cache](../originals/inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md). Focus on cached-decoder gate, using the task below to select the relevant subsection.

**External entry:** [SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). See [resource verification notes](../RESOURCES.md).

**Do:** Run a fixed token sequence through full and cached paths and explain any mismatch from the first divergent intermediate.

**Done when:** Do not proceed to quantization while unexplained baseline parity failures remain.

**Career block:** Run a full mock segment and audit the first two months; do not wait for course completion to apply. Use the [career reference](../originals/ai-career-acceleration/weeks/week-08.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-055.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Full-sequence and incremental outputs agree within declared tolerances. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-056"></a>

## Day 056 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Full-sequence and incremental outputs agree within declared tolerances.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
