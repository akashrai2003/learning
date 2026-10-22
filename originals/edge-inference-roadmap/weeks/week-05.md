# Week 05: CMake, testing, standard algorithms and checkpoint I/O

Days 029–035 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html): Follow the tutorial matching your installed stable CMake version; latest documentation can describe newer releases.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [Clang sanitizers](https://clang.llvm.org/docs/AddressSanitizer.html): Start with AddressSanitizer; linked guides explain other diagnostics and platform restrictions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 029 — CMake targets

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Source lists, libraries, executables, include paths and target-scoped settings.

**Build:** Create a core library, CLI and test target; set your own code to C++20 without changing upstream requirements.

**Verify:** A clean out-of-tree debug build and release build both work; document exact commands.

**Save:** code or configuration changes, any raw observations, and `notes/day-029.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 030 — Testing fundamentals

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Fixtures, assertions, boundary cases and independent expected results.

**Build:** Use a lightweight test framework or CTest executable; test shapes, dot products, reshape and parsing.

**Verify:** At least one intentionally broken implementation fails a meaningful test; tests are not copies of the implementation.

**Save:** code or configuration changes, any raw observations, and `notes/day-030.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 031 — Algorithms and lambdas

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Iterators, range algorithms, captures and comparator contracts.

**Build:** Replace appropriate manual loops with standard algorithms; implement a deterministic top-k reference.

**Verify:** Tied values have documented behavior; explain any reference captured by a stored lambda.

**Save:** code or configuration changes, any raw observations, and `notes/day-031.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 032 — Templates in moderation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Function templates, type deduction and compile-time constraints.

**Build:** Generalize one small utility for float and double; compare with an ordinary overload.

**Verify:** Both instantiations are tested; remove abstraction if it obscures ownership or error behavior.

**Save:** code or configuration changes, any raw observations, and `notes/day-032.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 033 — Binary checkpoint specification

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Byte order, magic, version, dtype, dimensions, offsets and lengths.

**Build:** Write a format specification before code; export two tensors from Python with explicit endian encoding.

**Verify:** A hand-inspected byte layout matches the spec; never serialize a compiler-dependent C++ struct directly.

**Save:** code or configuration changes, any raw observations, and `notes/day-033.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 034 — Defensive checkpoint reader

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Stream I/O, file size, checked offsets and validation before allocation.

**Build:** Read the tiny format in C++; reject invalid magic, unsupported version, truncation and inconsistent tensor sizes.

**Verify:** Generate separate corrupted fixtures; each fails before an unsafe read or unreasonable allocation.

**Save:** code or configuration changes, any raw observations, and `notes/day-034.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 035 — Clean-build checkpoint drill

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Reproducible builds and cross-language agreement.

**Build:** Build from a clean directory and load Python-exported tensors; compare every value and shape.

**Verify:** A peer could reproduce using only your README; capture compiler and dependency versions.

**Save:** code or configuration changes, any raw observations, and `notes/day-035.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
