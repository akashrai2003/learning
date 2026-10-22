# Week 06: Concurrency, Python bindings and systems foundations

Days 036–042 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines): Reference for resource management, ownership and concurrency. Read a handful of applicable rules after your own design.
- [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/): Focus on functions, NumPy buffers, lifetime and GIL rules; skip advanced binding features initially.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 036 — Threads and races

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Thread lifetime, join, shared state and data races.

**Build:** Write a two-thread counter example and a corrected version; keep the broken version in a separate test area.

**Verify:** Explain why apparent correct output does not prove race freedom; use ThreadSanitizer on a supported host build.

**Save:** code or configuration changes, any raw observations, and `notes/day-036.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 037 — Mutexes and condition variables

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Lock ownership, predicate waits, spurious wakeups and shutdown.

**Build:** Implement a bounded producer-consumer queue with a mutex and condition variables.

**Verify:** Empty waits, full waits and shutdown all terminate; test a waiting producer and consumer.

**Save:** code or configuration changes, any raw observations, and `notes/day-037.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 038 — Cancellation semantics

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Cooperative cancellation, request state and terminal events.

**Build:** Add request IDs and cancellation to a one-worker job executor; define queued versus running cancellation.

**Verify:** Each request emits exactly one terminal result; cancellation checks occur at documented safe points.

**Save:** code or configuration changes, any raw observations, and `notes/day-038.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 039 — Python-to-C++ boundary

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Bindings, array ownership, contiguous buffers and GIL concepts.

**Build:** Expose your dot product or Tensor operation using pybind11; reject unsupported dtype/layout rather than silently misreading it.

**Verify:** Noncontiguous input is rejected or explicitly copied; input owner stays alive during native access.

**Save:** code or configuration changes, any raw observations, and `notes/day-039.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 040 — Memory mapping and page faults

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Virtual addresses, resident pages, mmap, copy-on-write and file-backed storage.

**Build:** Add an optional read-only mapped checkpoint loader on your host; keep the stream loader for comparison.

**Verify:** Understand that mapped file size is not resident memory; validate offsets before producing views.

**Save:** code or configuration changes, any raw observations, and `notes/day-040.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 041 — Parallel work and false sharing

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Partitioned output, reductions, synchronization and cache lines.

**Build:** Parallelize independent rows of a matrix-vector operation with a simple fixed partition.

**Verify:** Matches single-thread output within tolerance; measure enough work to expose overhead and report a slower small case.

**Save:** code or configuration changes, any raw observations, and `notes/day-041.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 042 — Foundation gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Ownership, threads, binary validation and build reproducibility.

**Build:** Review your queue under stress and draw the ownership of model, session and temporary buffers.

**Verify:** No unsupported claim of production readiness; list unresolved races, lifetime risks and portability limits explicitly.

**Save:** code or configuration changes, any raw observations, and `notes/day-042.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
