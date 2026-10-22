# Week 12: INT8 quantization and quality evaluation

Days 078–084 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [ggml source and quantization implementation](https://github.com/ggml-org/llama.cpp/tree/master/ggml): Inspect format and kernel concepts after your scalar implementation; educational INT4 is not automatically a supported GGUF quant.
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 078 — Quantization arithmetic

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Scale, zero point, signed ranges, clipping and rounding.

**Build:** Implement scalar symmetric quantize/dequantize in Python with an explicit representable range.

**Verify:** Zero, boundaries, ties and outliers have specified behavior; test round-trip error.

**Save:** code or configuration changes, any raw observations, and `notes/day-078.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 079 — Granularity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Per-tensor, per-output-channel and group scales.

**Build:** Quantize one linear layer at each granularity and report error by output channel.

**Verify:** Scale indexing is tested; compare storage including scale overhead rather than raw integer data only.

**Save:** code or configuration changes, any raw observations, and `notes/day-079.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 080 — Native integer representation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Weight storage, accumulator width and overflow bounds.

**Build:** Write an INT8 loader and reference quantized matvec; keep the activation path explicitly documented.

**Verify:** Compare against dequantized reference; explain whether your first kernel uses integer or floating-point multiplication.

**Save:** code or configuration changes, any raw observations, and `notes/day-080.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 081 — Calibration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Representative input distribution and activation ranges.

**Build:** Collect layer statistics on a small public corpus; compare calibration sets from two domains.

**Verify:** No evaluation examples leak into tuning; report saturation and outlier effects.

**Save:** code or configuration changes, any raw observations, and `notes/day-081.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 082 — End-to-end quantized model

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Mixed precision, sensitive layers and integration.

**Build:** Replace selected projections with quantized versions; keep a float reference switch.

**Verify:** Layer and model numeric comparisons pass the chosen quality contract; quantify memory and latency.

**Save:** code or configuration changes, any raw observations, and `notes/day-082.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 083 — Task evaluation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Perplexity as one signal; application-level quality and failure cases.

**Build:** Create a fixed text evaluation subset and report float versus INT8 outcomes.

**Verify:** Keep tokenizer, prompts and decoding fixed; manually inspect examples where behavior changes.

**Save:** code or configuration changes, any raw observations, and `notes/day-083.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 084 — Quantization report

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Compression, performance and quality tradeoffs.

**Build:** Write a result table including scale overhead, artifact size, memory and task scores.

**Verify:** If INT8 is slower, explain likely unpacking or kernel overhead and retain the honest result.

**Save:** code or configuration changes, any raw observations, and `notes/day-084.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
