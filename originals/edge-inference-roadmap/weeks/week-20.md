# Week 20: Backend comparison and inference infrastructure

Days 134–140 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md): Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM): Optional comparison after your main runtime and application work; check supported models and packaging.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 134 — Upstream Snapdragon backend

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** CPU, Adreno and experimental Hexagon paths in the pinned runtime.

**Build:** Attempt the upstream Snapdragon path using its exact current instructions; enumerate actual devices.

**Verify:** Record what initializes and what runs; do not assume a GPU-looking offload flag proves GPU rather than another backend.

**Save:** code or configuration changes, any raw observations, and `notes/day-134.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 135 — Comparison contract

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Same-model comparisons versus whole-system comparisons.

**Build:** Define a matrix with runtime, quant, template, context, token counts and backend.

**Verify:** Mark unavoidable differences explicitly; avoid calling a different quantized model a hardware-only comparison.

**Save:** code or configuration changes, any raw observations, and `notes/day-135.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 136 — Automated matrix

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Configuration files, repetitions, timeout handling and result schema.

**Build:** Extend the Python runner to execute the small comparison matrix and resume incomplete experiments.

**Verify:** Failed runs remain visible and do not become zero latency; preserve raw logs.

**Save:** code or configuration changes, any raw observations, and `notes/day-136.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 137 — Second application runtime

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** LiteRT-LM architecture and model packaging.

**Build:** Read its official integration path; optionally run one supported sample if time remains.

**Verify:** Document tested versus untested behavior; do not expand scope if core comparisons remain incomplete.

**Save:** code or configuration changes, any raw observations, and `notes/day-137.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 138 — Request scheduling

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Single-user latency, bounded queues and admission control.

**Build:** Add one-worker queueing to your native service or app and measure queue time separately.

**Verify:** Concurrent requests cannot corrupt session state; reject overload predictably.

**Save:** code or configuration changes, any raw observations, and `notes/day-138.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 139 — Prefix reuse and cache validity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Exact token-prefix match, model identity and template identity.

**Build:** Implement a small cache-key experiment or simulation with explicit invalidation.

**Verify:** Changing model, template or token prefix prevents invalid reuse; measure benefit only for repeated prefixes.

**Save:** code or configuration changes, any raw observations, and `notes/day-139.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 140 — Deployment decision

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Latency, quality, memory, compatibility and sustained behavior.

**Build:** Choose a backend for the capstone using a decision table; include engineering complexity.

**Verify:** The choice follows your measurements; unsupported accelerators remain an optional branch.

**Save:** code or configuration changes, any raw observations, and `notes/day-140.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
