# Week 07: Python transformer mathematics and reference operations

Days 043–049 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 043 — Tensor layout and linear algebra

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Batch, sequence, hidden size, strides and matrix orientation.

**Build:** Write explicit-shape NumPy or PyTorch examples for linear projection and batched matmul.

**Verify:** Annotate every dimension and compare one tiny multiplication against a hand calculation.

**Save:** code or configuration changes, any raw observations, and `notes/day-043.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 044 — Stable softmax

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Exponent overflow, max subtraction, axis selection and masking.

**Build:** Implement stable softmax and causal masking without calling the high-level attention API.

**Verify:** Rows normalize correctly; large logits remain finite; define behavior for an all-masked row.

**Save:** code or configuration changes, any raw observations, and `notes/day-044.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 045 — Normalization

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** RMSNorm versus LayerNorm; epsilon and reduction precision.

**Build:** Implement RMSNorm with explicit dtype handling and compare to a trusted reference.

**Verify:** Test zero, constant and high-magnitude inputs; choose a numeric tolerance justified by dtype.

**Save:** code or configuration changes, any raw observations, and `notes/day-045.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 046 — Rotary positions

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Pair rotation, positions, frequencies and architecture conventions.

**Build:** Implement RoPE for a tiny tensor; isolate configuration and layout conventions.

**Verify:** Position zero behaves as expected; pair norms are preserved within rounding error; compare exact convention to the checkpoint.

**Save:** code or configuration changes, any raw observations, and `notes/day-046.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 047 — Attention reference

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Q, K, V, causal scores, scaling and head dimensions.

**Build:** Implement ordinary multi-head causal attention with explicit tensor operations.

**Verify:** Changing a future token cannot affect earlier outputs; a small example matches a manual or trusted computation.

**Save:** code or configuration changes, any raw observations, and `notes/day-047.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 048 — Feed-forward and residuals

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** SwiGLU, activation, gated projections and residual stream.

**Build:** Implement one compatible decoder block from your chosen architecture, including both residual paths.

**Verify:** Validate intermediate shapes and outputs; do not substitute architecture details without documenting them.

**Save:** code or configuration changes, any raw observations, and `notes/day-048.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 049 — Math reconstruction

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Forward-pass dependencies and numeric stability.

**Build:** Rebuild the block from a blank file, then compare it with the previous version.

**Verify:** Explain every matmul and reduction; save deterministic reference tensors for later C++ tests.

**Save:** code or configuration changes, any raw observations, and `notes/day-049.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
