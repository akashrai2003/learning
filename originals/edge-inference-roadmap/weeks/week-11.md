# Week 11: Profiling and the memory hierarchy

Days 071–077 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [Android Simpleperf](https://developer.android.com/ndk/guides/simpleperf): Use supported profiling modes and document permission limits; combine with your own stage timings.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 071 — Benchmark validity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Warmup, repetition, clocks, optimizer elimination and work accounting.

**Build:** Build a kernel benchmark with checksummed output and multiple input sizes.

**Verify:** Release builds do useful work; record raw observations and separate setup from the timed region.

**Save:** code or configuration changes, any raw observations, and `notes/day-071.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 072 — Profile before changing code

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Sampling versus instrumentation, symbols and call stacks.

**Build:** Profile host decode and annotate time in matvec, attention, allocation and other work.

**Verify:** Choose the largest actionable bottleneck using evidence; save a baseline profile.

**Save:** code or configuration changes, any raw observations, and `notes/day-072.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 073 — Allocation discipline

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Capacity reuse, arenas as a concept and lifetime-bound workspaces.

**Build:** Count or instrument allocations during decode; preallocate one repeated temporary buffer.

**Verify:** Correctness remains unchanged; report allocation count and latency before and after.

**Save:** code or configuration changes, any raw observations, and `notes/day-073.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 074 — Cache locality

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Contiguous traversal, strides and working-set size.

**Build:** Compare row and column traversal on matrices that fit and exceed cache.

**Verify:** Explain observed differences without attributing every result to bandwidth; repeat to check noise.

**Save:** code or configuration changes, any raw observations, and `notes/day-074.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 075 — Bandwidth and arithmetic intensity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Bytes moved, FLOPs, roofline estimates and batch-one decode.

**Build:** Estimate weight bytes read per token and achievable bandwidth; compare with measured decode.

**Verify:** State assumptions and explain why the simple bound is not a hardware guarantee.

**Save:** code or configuration changes, any raw observations, and `notes/day-075.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 076 — Prefill versus decode workload

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Matrix-matrix versus matrix-vector shapes; amortization.

**Build:** Benchmark several prompt lengths and output lengths separately.

**Verify:** Show that one tokens-per-second number cannot describe both phases; include context length in decode results.

**Save:** code or configuration changes, any raw observations, and `notes/day-076.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 077 — Optimization proposal

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Amdahl's law, confounds and experiment scope.

**Build:** Write three optimization hypotheses ranked by impact and effort; implement only the smallest justified change.

**Verify:** A report includes unchanged correctness tests and a confidence-limited interpretation of the measurements.

**Save:** code or configuration changes, any raw observations, and `notes/day-077.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
