# Week 13: INT4 packing, group quantization and fused dequantization

Days 085–091 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [ggml source and quantization implementation](https://github.com/ggml-org/llama.cpp/tree/master/ggml): Inspect format and kernel concepts after your scalar implementation; educational INT4 is not automatically a supported GGUF quant.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 085 — Bit operations

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Masks, shifts, sign extension and integer promotions.

**Build:** Implement encode/decode for signed 4-bit values; document your nibble ordering.

**Verify:** Exhaustively round-trip all representable values and both nibble positions; do not rely on ambiguous signed shifts.

**Save:** code or configuration changes, any raw observations, and `notes/day-085.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 086 — Packing arrays

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Odd lengths, padding and byte counts.

**Build:** Pack arbitrary-length arrays and record the original element count.

**Verify:** Lengths 0, 1, 2, 3 and large odd arrays pass; unused bits cannot produce a phantom value.

**Save:** code or configuration changes, any raw observations, and `notes/day-086.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 087 — Group quantization

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Group boundaries, scale storage and tail groups.

**Build:** Implement grouped INT4 export in Python and decode in C++.

**Verify:** Check dimensions not divisible by group size; include scale metadata in the file size.

**Save:** code or configuration changes, any raw observations, and `notes/day-087.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 088 — Reference INT4 matvec

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Unpack, rescale and accumulate without changing the model contract.

**Build:** Implement an intentionally simple correctness-first kernel.

**Verify:** Matches Python dequantized computation; document where quantization error enters.

**Save:** code or configuration changes, any raw observations, and `notes/day-088.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 089 — Fused dequantization

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Temporary buffers, repeated unpacking and data reuse.

**Build:** Compare pre-dequantized, per-call dequantized and inner-loop unpacking approaches.

**Verify:** Measure separately on small and large matrices; include temporary memory and setup time.

**Save:** code or configuration changes, any raw observations, and `notes/day-089.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 090 — Model experiment

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Sensitivity, group size and mixed precision choices.

**Build:** Run a small sweep of two group sizes and selected quantized layers.

**Verify:** Use the frozen task set and identical generation settings; do not optimize only perplexity.

**Save:** code or configuration changes, any raw observations, and `notes/day-090.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 091 — Quantization gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Compatibility versus educational formats.

**Build:** Write a precise format description and compare concepts with runtime-supported quants.

**Verify:** State that your INT4 format is educational and not automatically GGUF, QNN or hardware-native INT4.

**Save:** code or configuration changes, any raw observations, and `notes/day-091.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
