# Week 23: Evaluation, regression testing and sustained application behavior

Days 155–161 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): Optional reference for reproducible language-model task evaluation; your capstone still needs source-grounded custom tests.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.
- [SQLite documentation](https://www.sqlite.org/docs.html): Useful local metadata and index-manifest storage; vector search can start as your own exact scan.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 155 — Evaluation dataset

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Answerable, unanswerable, cross-passage and distracting evidence cases.

**Build:** Expand to approximately 50 carefully reviewed questions with gold sources and scoring rubrics.

**Verify:** Separate tuning examples from a held-out subset; retain source version and author review status.

**Save:** code or configuration changes, any raw observations, and `notes/day-155.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 156 — Retrieval evaluation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Recall at k, source relevance and failure categorization.

**Build:** Calculate retrieval metrics before generation; inspect misses by chunking and embedding cause.

**Verify:** Do not credit generation for evidence the retriever never supplied; report denominators.

**Save:** code or configuration changes, any raw observations, and `notes/day-156.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 157 — Answer and citation scoring

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Correctness, support, citation validity and appropriate abstention.

**Build:** Score answers with a human-readable rubric; use any model judge only as a secondary signal.

**Verify:** Manually audit disagreements and representative failures; distinguish correctness from fluent phrasing.

**Save:** code or configuration changes, any raw observations, and `notes/day-157.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 158 — Performance evaluation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Stage timings, percentile estimates and sustained runs.

**Build:** Run the fixed set under controlled conditions and preserve per-request records.

**Verify:** Small samples are not sold as reliable tail percentiles; include individual failures and warmup policy.

**Save:** code or configuration changes, any raw observations, and `notes/day-158.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 159 — Regression suite

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Frozen fixtures, acceptable drift and compatibility gates.

**Build:** Create release checks for tokenization, numeric parity, schema output and lifecycle behavior.

**Verify:** A deliberately changed template or bad model manifest is detected; tolerances have a documented reason.

**Save:** code or configuration changes, any raw observations, and `notes/day-159.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 160 — Ablation study

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** One variable at a time and interaction effects.

**Build:** Compare two context budgets or retrieval settings using the held-out protocol.

**Verify:** Report quality and latency together; avoid selecting a winner on the same examples used repeatedly for tuning.

**Save:** code or configuration changes, any raw observations, and `notes/day-160.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 161 — Evaluation gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Evidence-based claims and practical limits.

**Build:** Write a report with failures, sample sizes and the chosen operating point.

**Verify:** All tables can be regenerated from saved records; conclusions fit the measured scope.

**Save:** code or configuration changes, any raw observations, and `notes/day-161.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
