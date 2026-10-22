# Week 08: Checkpoint parity, tokenization and cached generation

Days 050–056 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.
- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers/index): Read the tokenizer pipeline and use official model tokenizer assets; a hand-written educational tokenizer is a separate exercise.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 050 — Architecture contract

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Config fields, tied embeddings, attention variants and checkpoint keys.

**Build:** Select one small supported checkpoint; map each tensor name and shape to your reference architecture.

**Verify:** Stop and resolve any mismatch rather than coercing shapes; record model revision and config.

**Save:** code or configuration changes, any raw observations, and `notes/day-050.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 051 — Checkpoint loading

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** State dictionaries, dtype conversion and inference mode.

**Build:** Load weights into the Python reference; compare the first block against the official implementation.

**Verify:** Use a fixed token-ID sequence and compare named intermediates; isolate the first layer that diverges.

**Save:** code or configuration changes, any raw observations, and `notes/day-051.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 052 — Full forward parity

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Final norm, output head and vocabulary logits.

**Build:** Run the full model for a short token sequence and compare logits.

**Verify:** Report max absolute error and selected logits; explain why argmax alone is insufficient validation.

**Save:** code or configuration changes, any raw observations, and `notes/day-052.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 053 — Tokenizer and chat template

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** BPE or model-specific tokenization, special tokens, BOS/EOS and templates.

**Build:** Use the official tokenizer; create fixed text-to-token fixtures including Unicode and whitespace.

**Verify:** Round trips and special-token behavior are documented; distinguish instruction template changes from model changes.

**Save:** code or configuration changes, any raw observations, and `notes/day-053.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 054 — KV cache implementation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Prefill, per-layer K/V storage, cache positions and context bounds.

**Build:** Add cache updates and one-token decode to your Python reference.

**Verify:** Cached and uncached logits agree for the same prefix within tolerance; reset actually clears session state.

**Save:** code or configuration changes, any raw observations, and `notes/day-054.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 055 — Sampling and sequence limits

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Greedy, temperature, seeded randomness, stop rules and logits processing.

**Build:** Implement greedy and basic temperature sampling; bound total context and generated tokens.

**Verify:** EOS and limit termination work; fixed-seed tests are reproducible within the controlled implementation.

**Save:** code or configuration changes, any raw observations, and `notes/day-055.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 056 — Reference freeze

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Model identity, fixtures and deterministic debug workloads.

**Build:** Freeze a small golden fixture set for operations, one block and full cached decode.

**Verify:** Write the exact numeric comparison contract; keep large weights out of your source repository.

**Save:** code or configuration changes, any raw observations, and `notes/day-056.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
