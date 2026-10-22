# Week 10: A complete educational C++ inference engine

Days 064–070 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.
- [llama2.c](https://github.com/karpathy/llama2.c): Read only after implementing your first forward pass. This is educational C, not a production C++ architecture template.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 064 — Whole-model execution

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Layer iteration, embedding lookup, final norm and output projection.

**Build:** Assemble the complete forward pass for token IDs using your validated blocks.

**Verify:** Compare final logits for a short fixed sequence; invalid token IDs fail cleanly.

**Save:** code or configuration changes, any raw observations, and `notes/day-064.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 065 — Native KV cache

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Layout, current position, capacity and session ownership.

**Build:** Implement fixed-capacity cache allocation and append; separate immutable model weights from mutable session state.

**Verify:** Two sessions do not share cache contents; boundary and reset cases match Python.

**Save:** code or configuration changes, any raw observations, and `notes/day-065.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 066 — Prefill and decode paths

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Full prefix versus incremental step and reusable workspace.

**Build:** Implement prefill and one-token decode functions with clearly timed boundaries.

**Verify:** Cache and uncached paths agree; avoid per-token workspace growth.

**Save:** code or configuration changes, any raw observations, and `notes/day-066.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 067 — Streaming and cancellation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Token events, completion events and cooperative stop checks.

**Build:** Expose a token callback and cancellation flag in the CLI; define callback thread and lifetime.

**Verify:** Exactly one terminal status follows completion or cancellation; callbacks cannot outlive their session.

**Save:** code or configuration changes, any raw observations, and `notes/day-067.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 068 — Tokenizer integration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** External token IDs first, compatible tokenizer second.

**Build:** Keep a token-ID debug mode and integrate a trusted tokenizer interface; preserve template fixtures.

**Verify:** End-to-end text prompt produces the same input IDs as the reference; Unicode output is not split into invalid UI fragments.

**Save:** code or configuration changes, any raw observations, and `notes/day-068.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 069 — Reference implementation reading

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Small C inference code versus your C++ structure.

**Build:** Read selected llama2.c forward and generation code after writing your own; note five design differences.

**Verify:** Explain correctness or ownership implications; do not paste an implementation you cannot reconstruct.

**Save:** code or configuration changes, any raw observations, and `notes/day-069.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 070 — Engine milestone

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Parity, bounds, errors and reproducibility.

**Build:** Run a clean end-to-end generation demo and write an architecture note with model/session/workspace ownership.

**Verify:** The tiny model works on host; report supported architecture and limitations rather than claiming general GGUF compatibility.

**Save:** code or configuration changes, any raw observations, and `notes/day-070.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
