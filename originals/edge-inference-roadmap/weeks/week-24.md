# Week 24: Release engineering, portfolio and the next specialization

Days 162–168 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Android application fundamentals](https://developer.android.com/guide/components/fundamentals): Review lifecycle and packaging requirements for your application. This curriculum is not a Play Store publishing checklist.
- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [llama.cpp repository and benchmark sources](https://github.com/ggml-org/llama.cpp): Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 162 — Model packaging

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Hashes, manifests, compatibility, licenses and asset versioning.

**Build:** Define a manifest and validate required files, runtime compatibility and hashes before activation.

**Verify:** A tampered asset is rejected; include model origin and required terms without redistributing restricted weights.

**Save:** code or configuration changes, any raw observations, and `notes/day-162.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 163 — Interrupted installation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Temporary downloads/copies, validation and atomic activation.

**Build:** Implement interrupted asset-install recovery using small fixtures before large files.

**Verify:** The old valid model remains usable; incomplete assets are never selected as active.

**Save:** code or configuration changes, any raw observations, and `notes/day-163.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 164 — Clean build and installation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Dependency pinning, reproducible instructions and device setup.

**Build:** Build from a clean directory and install using only documented steps.

**Verify:** Record any manual step you previously forgot; avoid claiming bit-for-bit reproducibility unless measured.

**Save:** code or configuration changes, any raw observations, and `notes/day-164.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 165 — Soak and lifecycle release check

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Repeated requests, cancellations, model switching and process restart.

**Build:** Run a bounded scripted reliability session and inspect memory plus errors.

**Verify:** Every failure is triaged; unresolved issues appear in known limitations rather than being hidden.

**Save:** code or configuration changes, any raw observations, and `notes/day-165.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 166 — Portfolio documentation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Architecture, tradeoffs, measurements and readable code.

**Build:** Write three READMEs for the educational engine, benchmark suite and capstone; include one optimization case study.

**Verify:** Another engineer can find the important code and reproduce a result without chat history.

**Save:** code or configuration changes, any raw observations, and `notes/day-166.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 167 — Upstream contribution preparation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Minimal reproduction, regression tests and project conventions.

**Build:** Prepare a small local patch, issue draft or documentation correction for a real observed problem.

**Verify:** No external message is sent automatically; your contribution is specific, reproducible and respectful of project scope.

**Save:** code or configuration changes, any raw observations, and `notes/day-167.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 168 — Final reconstruction and next plan

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Independent understanding and specialization choice.

**Build:** Explain the complete request path; reimplement one critical kernel from memory; choose ARM, NPU/compiler or runtime infrastructure for the next eight weeks.

**Verify:** Score the milestone rubric honestly; produce a concrete next-month plan based on the weakest demonstrated skill.

**Save:** code or configuration changes, any raw observations, and `notes/day-168.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
