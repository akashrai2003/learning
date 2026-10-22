# Week 09: C++ transformer kernels and checkpoint integration

Days 057–063 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 057 — Matrix-vector kernel

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Row-major layout, indexing and accumulation precision.

**Build:** Implement FP32 matvec against your TensorView; export random and adversarial fixtures from Python.

**Verify:** Test non-square shapes and mismatched dimensions; compare absolute and relative error.

**Save:** code or configuration changes, any raw observations, and `notes/day-057.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 058 — RMSNorm in C++

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Reduction, epsilon and output buffering.

**Build:** Port the Python RMSNorm with preallocated output and no hidden shape assumptions.

**Verify:** Match fixtures including zeros and extremes; document in-place aliasing support or reject aliasing.

**Save:** code or configuration changes, any raw observations, and `notes/day-058.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 059 — RoPE in C++

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Position indexing, frequency storage and rotation convention.

**Build:** Port the chosen model's RoPE; precompute only what the configuration requires.

**Verify:** Match Python across multiple positions; include a test that catches the wrong dimension pairing.

**Save:** code or configuration changes, any raw observations, and `notes/day-059.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 060 — Softmax and attention

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Stable exponentiation, causal bounds and score buffers.

**Build:** Implement scalar attention for one head then multiple heads; use explicit strides.

**Verify:** Match small reference tensors; verify future-token masking and finite outputs.

**Save:** code or configuration changes, any raw observations, and `notes/day-060.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 061 — Feed-forward and decoder block

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Intermediate buffers, activation and residual update order.

**Build:** Assemble one decoder block using your tested kernels.

**Verify:** Compare every stage with Python rather than debugging only final output; report first divergence.

**Save:** code or configuration changes, any raw observations, and `notes/day-061.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 062 — Checkpoint-to-native bridge

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Exported tensor names, config and endian-safe loading.

**Build:** Extend your educational format for the selected tiny checkpoint and load one complete block.

**Verify:** All shapes and lengths validated; no silent transpose; reject an unsupported architecture field.

**Save:** code or configuration changes, any raw observations, and `notes/day-062.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 063 — Block parity gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Numerical debugging and code review.

**Build:** Run all block fixtures, fix the earliest divergence and explain any tolerance change.

**Verify:** Do not move to full generation until block parity is established; list any model-specific assumptions.

**Save:** code or configuration changes, any raw observations, and `notes/day-063.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
