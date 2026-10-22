# Week 19: Qualcomm NPU deployment and vendor runtime constraints

Days 127–133 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [Qualcomm Genie LLM tutorial](https://github.com/qualcomm/ai-hub-apps/blob/release/tutorials/llm_on_genie/README.md): Target-aware vendor path; verify SDK, firmware and model-asset compatibility on your exact device.
- [ExecuTorch LLM documentation](https://docs.pytorch.org/executorch/stable/llm/working-with-llms.html): Start with one supported example. Exporter, runtime and backend versions must agree.
- [llama.cpp Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md): Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 127 — Choose one vendor route

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Genie versus ExecuTorch Qualcomm backend; SDK and target compatibility.

**Build:** Select one supported official example and record required SDK, device family and assets.

**Verify:** Confirm documentation applies to your target or state the gap; do not mix incompatible release instructions.

**Save:** code or configuration changes, any raw observations, and `notes/day-127.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 128 — Prepare reproducible assets

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Model revision, compiler options and target-specific artifacts.

**Build:** Obtain or compile supported model assets; hash every configuration and binary used.

**Verify:** A manifest links each artifact to its source and target; respect model terms and keep credentials out of records.

**Save:** code or configuration changes, any raw observations, and `notes/day-128.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 129 — First target execution

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Runtime libraries, initialization and diagnostic logs.

**Build:** Run the smallest supported example; capture complete initialization and execution logs.

**Verify:** Classify errors as asset, ABI, library, operator or device compatibility failures.

**Save:** code or configuration changes, any raw observations, and `notes/day-129.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 130 — Numerical validation

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Reference outputs and preprocessing equivalence.

**Build:** Compare vendor-run output with the appropriate reference using fixed token IDs or tensors.

**Verify:** Tokenizer and model configuration agree; task-level behavior meets a stated threshold.

**Save:** code or configuration changes, any raw observations, and `notes/day-130.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 131 — Prefill and decode configuration

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Supported sequence shapes, cache layout and preparation cost.

**Build:** Vary only documented options for one model; record first-use and repeated-run behavior.

**Verify:** Do not infer unrestricted dynamic shapes from one working sequence; list actual supported configurations.

**Save:** code or configuration changes, any raw observations, and `notes/day-131.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 132 — Partitioning and overhead

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Host work, accelerator work, transfers and synchronization.

**Build:** Inspect profiling or logs for fallback and transfer boundaries; estimate their contribution only where observable.

**Verify:** Separate measured timings from inferred bottlenecks; save raw traces.

**Save:** code or configuration changes, any raw observations, and `notes/day-132.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 133 — Vendor path gate

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Practical support and reproducibility.

**Build:** Write a setup guide or a high-quality failure report with a CPU fallback path.

**Verify:** The roadmap continues even if OEM firmware blocks NPU access; no root requirement is invented.

**Save:** code or configuration changes, any raw observations, and `notes/day-133.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
