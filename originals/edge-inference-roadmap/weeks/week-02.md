# Week 02: Control flow, numeric types and reproducible experiments

Days 008–014 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md): Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 008 — Conditions and validation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** if, else, booleans, comparison, short-circuit evaluation.

**Build:** Add model-size and quantization validation to the estimator; separate validation from output.

**Verify:** Zero, negative, absurdly large and malformed input receive deliberate behavior; explain each branch.

**Save:** code or configuration changes, any raw observations, and `notes/day-008.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 009 — Loops and running statistics

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** for and while loops; counters and accumulation.

**Build:** Write a C++ latency-summary program over a fixed array; calculate count, min, max and arithmetic mean.

**Verify:** Check an empty list and a single value; compare against Python and explain why an average hides outliers.

**Save:** code or configuration changes, any raw observations, and `notes/day-009.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 010 — Floating-point behavior

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Float versus double, rounding, relative and absolute error.

**Build:** Create cancellation and summation examples; implement a numeric comparison helper.

**Verify:** Demonstrate why exact equality fails for some calculations; use a scale-aware tolerance with a near-zero case.

**Save:** code or configuration changes, any raw observations, and `notes/day-010.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 011 — Integers and size arithmetic

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Fixed-width integers, size_t, signedness and checked multiplication.

**Build:** Rework byte-size estimation with checked integer operations; retain a separate display conversion.

**Verify:** Oversized dimensions fail before wraparound; explain why casting after overflow cannot repair it.

**Save:** code or configuration changes, any raw observations, and `notes/day-011.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 012 — Strings and configuration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** std::string, parsing and error states; avoid dangling views for now.

**Build:** Parse a simple command line for model label, context and threads; keep options few and documented.

**Verify:** Unknown flags and missing values fail clearly; quoted paths are handled by the calling shell or argument list.

**Save:** code or configuration changes, any raw observations, and `notes/day-012.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 013 — Small baseline matrix

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Prefill versus decode, thread counts, fresh versus reused process.

**Build:** Extend the Python runner to three prompt lengths and two thread counts with three repetitions each.

**Verify:** Keep actual input/output counts; report individual observations plus median; do not equate fresh process with cold filesystem cache.

**Save:** code or configuration changes, any raw observations, and `notes/day-013.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 014 — Review and measurement audit

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Distinguish load, tokenization, queueing, prefill and first-token display.

**Build:** Draw the timing boundaries in prose; rewrite one function from memory and inspect every baseline log.

**Verify:** Identify one confound such as charging or prompt-template variation; correct the next experiment plan.

**Save:** code or configuration changes, any raw observations, and `notes/day-014.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
