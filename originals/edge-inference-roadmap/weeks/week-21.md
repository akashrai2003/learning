# Week 21: Offline document assistant: retrieval and resource ownership

Days 141–147 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [SQLite documentation](https://www.sqlite.org/docs.html): Useful local metadata and index-manifest storage; vector search can start as your own exact scan.
- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [SmolLM2-135M official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 141 — Capstone scope

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Offline requirements, public data, task definition and budgets.

**Build:** Choose a small public corpus and 20 initial questions; define target memory, latency and quality envelopes.

**Verify:** Budgets are provisional hypotheses; avoid importing sensitive work documents.

**Save:** code or configuration changes, any raw observations, and `notes/day-141.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 142 — Text ingestion

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Encoding, document identity, page/source metadata and incremental updates.

**Build:** Implement text or pre-extracted PDF text import with stable source IDs.

**Verify:** Every chunk can be traced to a source location; invalid text input gets an explicit result.

**Save:** code or configuration changes, any raw observations, and `notes/day-142.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 143 — Chunking baseline

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Boundaries, overlap, token budget and provenance.

**Build:** Implement deterministic chunking and fixtures for headings, short sections and long paragraphs.

**Verify:** No dropped or duplicated source spans beyond deliberate overlap; metadata survives splitting.

**Save:** code or configuration changes, any raw observations, and `notes/day-143.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 144 — Embedding baseline

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Normalization, similarity choice and model residency.

**Build:** Deploy a small supported embedding model or precompute embeddings on host for the first baseline.

**Verify:** Label precomputed versus fully on-device stages; compare identical text preprocessing.

**Save:** code or configuration changes, any raw observations, and `notes/day-144.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 145 — Exact retrieval

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Cosine or dot similarity, top-k, stable ties and index persistence.

**Build:** Implement exact vector search over the small corpus and a persisted index manifest.

**Verify:** Compare against a Python oracle; query and index embedding versions must match.

**Save:** code or configuration changes, any raw observations, and `notes/day-145.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 146 — Answer assembly

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Prompt budget, source labels, truncation and abstention.

**Build:** Build a retrieved-context prompt with a strict token budget and requested citations.

**Verify:** Long retrieval output cannot overflow the session; missing evidence yields an abstention path.

**Save:** code or configuration changes, any raw observations, and `notes/day-146.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 147 — RAG baseline gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Traceability and stage-level measurements.

**Build:** Run the first end-to-end questions offline and save retrieval plus answer traces.

**Verify:** Report retrieval, prompt construction, prefill and decode separately; inspect cited sources manually.

**Save:** code or configuration changes, any raw observations, and `notes/day-147.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
