# Week 22: Capstone: memory management, quality and optional multimodality

Days 148–154 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [SQLite documentation](https://www.sqlite.org/docs.html): Useful local metadata and index-manifest storage; vector search can start as your own exact scan.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 148 — Residency policy

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Embedding and generator memory, sequential loading and peak allocations.

**Build:** Compare keeping both models resident with loading one on demand.

**Verify:** Measure transition latency and available memory; choose a policy based on the phone rather than model file sizes.

**Save:** code or configuration changes, any raw observations, and `notes/day-148.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 149 — Session budgeting

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Context occupancy, source compression and history limits.

**Build:** Implement bounded history and explicit source selection; keep provenance after truncation.

**Verify:** No hidden loss of system instructions or source labels; test a full-context request.

**Save:** code or configuration changes, any raw observations, and `notes/day-149.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 150 — Index update integrity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Versioning, crash recovery and atomic replacement.

**Build:** Build a new index separately, validate it, then activate it atomically where supported.

**Verify:** Interrupted updates leave the previous valid index usable; reject embedding-version mismatch.

**Save:** code or configuration changes, any raw observations, and `notes/day-150.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 151 — Structured extraction task

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Schema validation, evidence spans and unsupported claims.

**Build:** Add a narrow document extraction task such as dates and identifiers with citations.

**Verify:** Validate schema and source evidence; invalid output cannot silently become trusted data.

**Save:** code or configuration changes, any raw observations, and `notes/day-151.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 152 — Optional audio or vision branch

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Preprocessing, encoder cost, streaming and model scheduling.

**Build:** Choose one tiny supported speech or vision example; if core work is unfinished, use the session for debugging instead.

**Verify:** Measure preprocessing and inference separately; label this as optional and never block the text capstone.

**Save:** code or configuration changes, any raw observations, and `notes/day-152.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 153 — Failure injection

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Corrupt assets, cancellation, missing index and invalid configuration.

**Build:** Create a small failure matrix and exercise recovery paths in the app.

**Verify:** Each failure has a useful user-visible result and a diagnostic log without private content.

**Save:** code or configuration changes, any raw observations, and `notes/day-153.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 154 — Resource-policy gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Quality versus memory and latency.

**Build:** Write a policy note covering residency, budgets, queueing and model choice.

**Verify:** Explain the worst observed failure and what remains unmeasured; do not claim reliability from one happy-path demo.

**Save:** code or configuration changes, any raw observations, and `notes/day-154.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
