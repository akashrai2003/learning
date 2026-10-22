# Week 01: First C++ programs and an Android baseline

Days 001–007 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [C++ language curriculum](https://www.learncpp.com/): Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.
- [Android NDK](https://developer.android.com/ndk/guides): Primary reference for native builds, ABI, native APIs and tooling.
- [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md): Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 001 — Development inventory

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Compiler versus interpreter; source, executable, host and target.

**Build:** Create a learning repository; record phone model, real RAM, Android version, available storage, computer compiler and Python versions; compile a hello-world C++ program.

**Verify:** Run the executable from a terminal and explain every command; distinguish physical RAM from RAM expansion.

**Save:** code or configuration changes, any raw observations, and `notes/day-001.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 002 — Variables and initialization

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Basic types, initialization, cout, cin, statements and compiler diagnostics.

**Build:** Write a CLI that asks for parameter count and bit width and estimates raw weight bytes; use double for the first version.

**Verify:** Check 1B at 4 bits gives 500 million decimal bytes; distinguish GB from GiB; reject a negative input.

**Save:** code or configuration changes, any raw observations, and `notes/day-002.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 003 — Functions and parameters

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Return types, local scope, pass by value and function declarations.

**Build:** Split the estimator into conversion, estimation and presentation functions; add KV and runtime allowance as explicit user inputs.

**Verify:** Explain why the result is an estimate rather than phone capacity; manually check three configurations.

**Save:** code or configuration changes, any raw observations, and `notes/day-003.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 004 — Phone development connection

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** ADB, developer options, host-target separation; upstream Android build guide.

**Build:** Enable debugging, connect by USB, inspect device properties and run a harmless shell command; choose one small model and record its origin and terms.

**Verify:** Save device metadata and the exact model identity; do not claim an accelerator works before a model runs.

**Save:** code or configuration changes, any raw observations, and `notes/day-004.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 005 — First native baseline

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Runtime commit, release build, model hash, CPU execution and context limits.

**Build:** Follow upstream instructions for the pinned checkout; deploy a CPU executable and run one short prompt; save the complete command and logs.

**Verify:** Confirm generation occurs on the phone and backend logs identify CPU; record actual tokens rather than word counts.

**Save:** code or configuration changes, any raw observations, and `notes/day-005.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 006 — Simple Python experiment runner

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** subprocess argument lists, timeouts, JSON records and error capture.

**Build:** Write an ADB wrapper for one inference run; capture stdout, stderr, return code, elapsed time and configuration.

**Verify:** Disconnect the device or use a bad path and confirm a structured failure record; do not hide errors.

**Save:** code or configuration changes, any raw observations, and `notes/day-006.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 007 — Week-one reconstruction

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Recall variables, functions and the host-target build path.

**Build:** Rewrite the weight estimator without notes; reproduce one phone run; write a one-page explanation of what you measured.

**Verify:** A fresh shell can reproduce your command; list three unknowns and choose one to resolve next week.

**Save:** code or configuration changes, any raw observations, and `notes/day-007.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
