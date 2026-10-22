# Week 14: ARM NEON, threading and measured optimization

Days 092–098 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Arm Neon intrinsics reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html): Look up exact intrinsics and required architecture features; preserve a scalar reference and correct tail handling.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 092 — Vectorization fundamentals

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** SIMD lanes, loads, reductions, alignment and tails.

**Build:** Inspect compiler vectorization for your scalar dot product on the target.

**Verify:** Save compiler diagnostics or assembly; identify vector and scalar remainder paths.

**Save:** code or configuration changes, any raw observations, and `notes/day-092.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 093 — First NEON kernel

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** ARM intrinsics for vector multiply-add and reduction.

**Build:** Implement one FP32 dot-product kernel with a scalar fallback.

**Verify:** Test odd lengths, tiny lengths and unaligned starting offsets where allowed; compare numerical error.

**Save:** code or configuration changes, any raw observations, and `notes/day-093.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 094 — Quantized kernel experiment

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Packed data, unpack cost and accumulation.

**Build:** Optimize one INT8 or INT4 inner loop only after the reference passes.

**Verify:** Compare output and throughput across representative shapes; keep unsupported instruction paths disabled.

**Save:** code or configuration changes, any raw observations, and `notes/day-094.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 095 — Thread count and partitioning

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Work size, scheduling overhead and heterogeneous cores.

**Build:** Sweep a few thread counts with fixed workloads and supported affinity experiments if available.

**Verify:** Do not assume all cores maximize performance; report sustained results and permitted controls.

**Save:** code or configuration changes, any raw observations, and `notes/day-095.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 096 — False sharing and buffers

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Cache-line sharing, independent output and synchronization.

**Build:** Construct a controlled false-sharing example, then revise data placement.

**Verify:** Explain why the demonstration may vary with hardware; never treat padding as a universal optimization.

**Save:** code or configuration changes, any raw observations, and `notes/day-096.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 097 — Fusion or reuse experiment

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Removing a temporary or repeated computation.

**Build:** Try one small fusion or workspace reuse based on profiling; retain reference operation tests.

**Verify:** Measure total decode impact, not just an isolated hot-loop speedup.

**Save:** code or configuration changes, any raw observations, and `notes/day-097.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 098 — Optimization milestone

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Performance claims and reproducible evidence.

**Build:** Publish a local report with one successful optimization and one failed hypothesis.

**Verify:** Every claimed improvement includes workload, build, repetitions and correctness evidence.

**Save:** code or configuration changes, any raw observations, and `notes/day-098.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
