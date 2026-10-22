# Week 04: Classes, RAII, moves and a minimal tensor

Days 022–028 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines): Reference for resource management, ownership and concurrency. Read a handful of applicable rules after your own design.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 022 — Structs and invariants

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Members, aggregate initialization, encapsulation and class invariants.

**Build:** Create a Shape type with checked element count and a documented policy on zero dimensions.

**Verify:** Invalid shapes cannot silently allocate a wrapped size; unit tests cover rank and zero policy.

**Save:** code or configuration changes, any raw observations, and `notes/day-022.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 023 — Constructors and RAII

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Initialization lists, destructors, resource acquisition and cleanup.

**Build:** Create an owning FP32 Tensor using vector storage and Shape; provide checked indexing.

**Verify:** Construction establishes a valid state; throwing validation does not leak resources.

**Save:** code or configuration changes, any raw observations, and `notes/day-023.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 024 — Copy and move semantics

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Value semantics, references, move operations and moved-from states.

**Build:** Instrument a small demonstration type; compare copying Tensor with moving it; prefer the rule of zero where possible.

**Verify:** Predict the observed operations before running; explain why std::move does not itself move bytes.

**Save:** code or configuration changes, any raw observations, and `notes/day-024.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 025 — Smart pointers

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** unique_ptr, shared_ptr, ownership graphs and cycles.

**Build:** Wrap a small opaque resource in unique ownership; design session and model ownership on paper.

**Verify:** No shared_ptr is added without a shared-ownership reason; destruction order is testable.

**Save:** code or configuration changes, any raw observations, and `notes/day-025.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 026 — Views and reshaping

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Owning tensors versus borrowed views; contiguity and shape compatibility.

**Build:** Implement TensorView over a span and reshape for contiguous tensors; document invalidation.

**Verify:** Reject mismatched element counts; test views while the owner is stable and demonstrate the invalidation hazard separately.

**Save:** code or configuration changes, any raw observations, and `notes/day-026.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 027 — Enums and errors

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** enum class, error categories, exceptions and alternative return values.

**Build:** Define shape, dtype and load errors; make CLI boundaries convert failures into readable diagnostics.

**Verify:** A malformed operation reports the right category; no catch-all silently converts a failure into a valid tensor.

**Save:** code or configuration changes, any raw observations, and `notes/day-027.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 028 — Tensor design review

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Invariants, rule of zero, copies, borrows and failure paths.

**Build:** Write a short API document and simplify your implementation; reconstruct TensorView without reference.

**Verify:** Explain who owns each byte and when each view becomes invalid; all tensor tests pass.

**Save:** code or configuration changes, any raw observations, and `notes/day-028.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
