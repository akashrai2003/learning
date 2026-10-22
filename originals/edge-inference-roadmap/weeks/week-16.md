# Week 16: Kotlin, JNI and a minimal offline application

Days 106–112 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [Android JNI tips](https://developer.android.com/training/articles/perf-jni): Read ownership, references, threads and exception guidance before integrating native generation.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 106 — Kotlin essentials

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** val/var, functions, classes, nullability and basic collections.

**Build:** Build a tiny Android screen displaying selected model metadata; avoid complex UI architecture.

**Verify:** Understand every Kotlin statement used; the app installs and runs without inference dependencies.

**Save:** code or configuration changes, any raw observations, and `notes/day-106.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 107 — JNI boundary design

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Opaque handles, ownership, native exceptions and API contracts.

**Build:** Specify create, load, generate, cancel, reset and destroy operations before implementing them.

**Verify:** Document valid call sequences and invalid handle behavior; no C++ exception crosses JNI unchecked.

**Save:** code or configuration changes, any raw observations, and `notes/day-107.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 108 — Native library integration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Shared-library packaging, ABI selection and library loading.

**Build:** Load a native library and call a small metadata or dot-product function from Kotlin.

**Verify:** Verify the packaged ABI matches the phone; failure to load is visible and diagnosable.

**Save:** code or configuration changes, any raw observations, and `notes/day-108.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 109 — Background execution

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** UI thread, worker ownership and nonblocking callbacks.

**Build:** Run a simulated token stream on a background worker and display updates.

**Verify:** UI remains responsive; callback ordering and worker lifetime are documented.

**Save:** code or configuration changes, any raw observations, and `notes/day-109.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 110 — Real streaming inference

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** JNI string/array conversion, token bytes and buffering.

**Build:** Connect the runtime and stream generated text, preserving valid UTF-8 assembly.

**Verify:** A non-ASCII prompt and output do not corrupt text; first-token display timing is measured at the UI boundary.

**Save:** code or configuration changes, any raw observations, and `notes/day-110.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 111 — Stop and cleanup

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Cooperative stop, session state and resource release.

**Build:** Implement stop, reset and model release with explicit state transitions.

**Verify:** Rapid start/stop cannot use a freed handle; exactly one terminal status reaches the UI.

**Save:** code or configuration changes, any raw observations, and `notes/day-111.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 112 — Application milestone

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Minimal surface and observable behavior.

**Build:** Demonstrate prompt, streaming output, stop and metadata in airplane mode.

**Verify:** No network service is needed for inference after assets are installed; keep logs of a complete request lifecycle.

**Save:** code or configuration changes, any raw observations, and `notes/day-112.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
