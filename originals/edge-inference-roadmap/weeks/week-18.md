# Week 18: ExecuTorch export, graph lowering and delegation

Days 120–126 • Six 150-minute sessions plus one 120-minute review • Optional extensions up to three hours per day.

## Reading for this week

- [ExecuTorch LLM documentation](https://docs.pytorch.org/executorch/stable/llm/working-with-llms.html): Start with one supported example. Exporter, runtime and backend versions must agree.
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html): Use tensor, inference-mode and export API references matching your installed release.

Read the subsection matching the daily study prompt. The assignment is original curriculum design; linked sources explain APIs and concepts rather than supplying a prescribed solution.

## Daily sessions

### Day 120 — Export concepts

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Eager execution, captured graphs, shape constraints and graph breaks.

**Build:** Export a tiny linear model with torch.export and inspect the graph.

**Verify:** Explain parameters, inputs, outputs and shape guards; pin compatible package versions.

**Save:** code or configuration changes, any raw observations, and `notes/day-120.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 121 — Small-model deployment

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Portable runtime artifact and C++ runner.

**Build:** Follow a minimal ExecuTorch tutorial before an LLM example; deploy a tiny supported model.

**Verify:** Compare native outputs against Python; capture serialization and runtime versions.

**Save:** code or configuration changes, any raw observations, and `notes/day-121.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 122 — Operator coverage

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Supported ops, decomposition and backend requirements.

**Build:** Inspect an encoder or small transformer subgraph and list required operations.

**Verify:** Identify unsupported ops before lengthy conversion; distinguish export failure from execution failure.

**Save:** code or configuration changes, any raw observations, and `notes/day-122.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 123 — Quantized export

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Calibration, operator-level quantization and numerical checking.

**Build:** Quantize and export a small supported model following the selected backend recipe.

**Verify:** Record scale/dtype decisions and output drift; avoid assuming your educational INT4 is accepted.

**Save:** code or configuration changes, any raw observations, and `notes/day-123.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 124 — Delegation and fallback

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Partitioned graphs, data transfer and backend logs.

**Build:** Run a documented delegate example and inspect what was delegated.

**Verify:** Save evidence of actual backend execution; a successful API call alone is not proof of NPU use.

**Save:** code or configuration changes, any raw observations, and `notes/day-124.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 125 — Documented LLM attempt

**Time:** 150 minutes: 15 recall, 30 study, 75 implementation, 20 verification, 10 logging.

**Study:** Known-supported model and end-to-end recipe.

**Build:** Attempt one pinned official LLM workflow with a predeclared time budget; preserve each failure stage.

**Verify:** If blocked, provide a minimal reproducible issue and retain the small-model success as the milestone.

**Save:** code or configuration changes, any raw observations, and `notes/day-125.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Use up to 30 extra minutes for one additional edge case or a closed-book reconstruction of today's central function. Avoid starting an unrelated feature.

### Day 126 — Export milestone

**Time:** 120 minutes: 20 recall, 60 repair/reconstruction, 25 milestone review, 15 planning.

**Study:** Compatibility matrix and evidence.

**Build:** Write model, shape, dtype, target and delegated-op compatibility tables.

**Verify:** Separate verified support, documented-but-untested support and failed experiments.

**Save:** code or configuration changes, any raw observations, and `notes/day-126.md` using the daily-log template. Record unfinished work honestly.

**Optional extension:** Spend up to 60 additional minutes closing the most important failed check. If all checks pass, teach the week's core idea aloud and write next week's first failing test.

## Weekly evidence

- One working or explicitly diagnosed implementation.
- An independent correctness check or concrete failure reproduction.
- A short explanation of ownership, numerics or measurement relevant to this week.
- A list of open issues and the next action for each.

[Back to navigation](../02-week-index.md) · [Milestone rubric](../guides/milestones.md)
