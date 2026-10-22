# Week 03: References, pointers, containers and debugging

Days 015–021 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [Clang sanitizers](https://clang.llvm.org/docs/AddressSanitizer.html): Start with AddressSanitizer; linked guides explain other diagnostics and platform restrictions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 015 — References and const

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Object identity, aliases, const references and parameter contracts.

**Build:** Implement vector sum and normalization with explicit input/output ownership; compare pass-by-value and const-reference signatures.

**Verify:** Predict which calls copy data; explain why const is an interface property and not a thread-safety guarantee.

**Save:** code or configuration changes, any raw observations, and `notes/day-015.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 016 — Pointers and lifetimes

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Addresses, dereference, null, object lifetime and storage duration.

**Build:** Write tiny safe pointer examples; deliberately inspect a dangling pointer example in an isolated failing test.

**Verify:** Explain precisely when the pointee dies; do not run known undefined behavior in normal application code.

**Save:** code or configuration changes, any raw observations, and `notes/day-016.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 017 — std::vector

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Size, capacity, contiguous storage and iterator invalidation.

**Build:** Build a vector-based dot product with dimension validation; inspect capacity changes during append.

**Verify:** Test mismatched sizes and zero length; explain why reallocation can invalidate pointers and references.

**Save:** code or configuration changes, any raw observations, and `notes/day-017.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 018 — std::array and spans

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Fixed extent versus dynamic containers; non-owning std::span.

**Build:** Implement dot-product and min/max functions taking spans so array and vector callers share code.

**Verify:** Both owners can call the same function; document that a span does not extend the underlying lifetime.

**Save:** code or configuration changes, any raw observations, and `notes/day-018.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 019 — Debugger workflow

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Breakpoints, step into/over, stack frames and variable inspection.

**Build:** Introduce an indexing bug in a small program and find it with GDB or LLDB before reading the answer.

**Verify:** Save a short debugging transcript showing the first wrong state, not merely the final crash.

**Save:** code or configuration changes, any raw observations, and `notes/day-019.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 020 — Sanitizer workflow

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** AddressSanitizer and UndefinedBehaviorSanitizer; debug symbols.

**Build:** Build isolated out-of-bounds and use-after-free tests; repair them and add regression cases.

**Verify:** The tools diagnose the intentional failures where supported; repaired normal tests run clean; keep unsafe demos separate.

**Save:** code or configuration changes, any raw observations, and `notes/day-020.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 021 — Closed-book container challenge

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Pointers, spans, lifetime, resizing and const.

**Build:** Implement cosine similarity from an empty file with explicit zero-vector behavior; review all borrowed views.

**Verify:** Match a Python oracle and explain every container choice; name the lifetime assumptions in the API.

**Save:** code or configuration changes, any raw observations, and `notes/day-021.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
