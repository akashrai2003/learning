# Week 15: Android NDK deployment and native diagnostics

Days 099–105 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [Android Simpleperf](https://developer.android.com/ndk/guides/simpleperf): Use supported profiling modes and document permission limits; combine with your own stage timings.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 099 — Cross-compilation fundamentals

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Target triple, ABI, API level, libc and host tools.

**Build:** Create an Android CMake configuration for your educational engine; retain host tests.

**Verify:** The same source builds for host and arm64-v8a with separate build directories.

**Save:** code or configuration changes, any raw observations, and `notes/day-099.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 100 — Deploy the tiny engine

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Executable permissions, library paths and file locations.

**Build:** Push the native CLI and tiny checkpoint through ADB and run fixed token-ID input.

**Verify:** On-phone logits match the host within justified tolerance; distinguish ISA-related numeric differences from bugs.

**Save:** code or configuration changes, any raw observations, and `notes/day-100.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 101 — Symbols and crashes

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Debug symbols, backtraces and address-to-line mapping.

**Build:** Create an isolated debug crash and symbolize it; repair the bug.

**Verify:** The trace identifies your source function; do not distribute the intentional crash in the app.

**Save:** code or configuration changes, any raw observations, and `notes/day-101.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 102 — Android logs and errors

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** logcat, native error propagation and app-visible diagnostics.

**Build:** Add structured stage logs without logging document contents by default.

**Verify:** A missing checkpoint and unsupported format generate useful diagnostics and a nonzero CLI status.

**Save:** code or configuration changes, any raw observations, and `notes/day-102.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 103 — Native profiling

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Simpleperf permissions, sample collection and symbol resolution.

**Build:** Profile a debuggable or otherwise supported native target; fall back to scoped timing if restricted.

**Verify:** State which counters and traces are actually available; never fabricate unavailable hardware measurements.

**Save:** code or configuration changes, any raw observations, and `notes/day-103.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 104 — Memory accounting

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** RSS/PSS, file mappings, shared pages and allocator behavior.

**Build:** Collect available process memory information while loading, prefilling and decoding.

**Verify:** Explain why mapped bytes, RSS and PSS differ; preserve raw tool output alongside parsed values.

**Save:** code or configuration changes, any raw observations, and `notes/day-104.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 105 — Device parity gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Host-target reproducibility and compatibility.

**Build:** Write exact build, deploy and run instructions for the pinned versions.

**Verify:** Recreate the run from a clean shell; document ABI, Android version, model revision and backend.

**Save:** code or configuration changes, any raw observations, and `notes/day-105.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
