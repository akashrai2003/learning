# Week 17: Lifecycle, sustained performance and resource policies

Days 113–119 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [Android Simpleperf](https://developer.android.com/ndk/guides/simpleperf): Use supported profiling modes and document permission limits; combine with your own stage timings.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 113 — Activity recreation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** UI state versus native engine ownership.

**Build:** Test rotation or recreation during generation; decide where the engine lifetime belongs.

**Verify:** No duplicate generation or orphaned callback; explain behavior instead of suppressing lifecycle events blindly.

**Save:** code or configuration changes, any raw observations, and `notes/day-113.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 114 — Background and process loss

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Background limits, process death and recoverable state.

**Build:** Define whether generation pauses, cancels or continues only under permitted app behavior; persist lightweight metadata.

**Verify:** Relaunch is coherent after process termination; do not promise survival without the required Android execution model.

**Save:** code or configuration changes, any raw observations, and `notes/day-114.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 115 — Model switching

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Unloading order, outstanding requests and memory peaks.

**Build:** Implement sequential model switching or simulate it if only one asset is available.

**Verify:** Reject or queue new requests during switch; no callback references unloaded state.

**Save:** code or configuration changes, any raw observations, and `notes/day-115.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 116 — Sustained workload design

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Thermal state, charging, ambient conditions and run order.

**Build:** Run a 10-15 minute sequence with repeated fixed requests; log throughput and thermal status when available.

**Verify:** Plot performance over time; report test conditions and any throttling indicators without inferring unsupported sensor values.

**Save:** code or configuration changes, any raw observations, and `notes/day-116.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 117 — Energy measurement limits

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Power versus energy, battery gauge noise and external meters.

**Build:** Define an energy experiment; use coarse battery deltas only as exploratory evidence if no reliable meter exists.

**Verify:** Do not claim precise joules per token from a brief percentage change; state measurement uncertainty.

**Save:** code or configuration changes, any raw observations, and `notes/day-117.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 118 — Resource-budget policy

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Queue size, context cap, token budget and memory-pressure response.

**Build:** Implement conservative limits and a clear out-of-budget message.

**Verify:** Over-limit requests fail before excessive allocation; document how limits were chosen from measurements.

**Save:** code or configuration changes, any raw observations, and `notes/day-118.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 119 — Reliability gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Repeated cancellation, load/unload and state transitions.

**Build:** Run a structured lifecycle soak test with failure injection and raw logs.

**Verify:** No uncontrolled memory growth in available measurements; list remaining platform restrictions.

**Save:** code or configuration changes, any raw observations, and `notes/day-119.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
