# Week 21: Online softmax and efficient attention

Days 141–147 · 17 hours total · 14 technical + 2 career + 1 review hour.

**Weekly output:** Tiled attention prototype and attention IO note.

**Prerequisite:** Pass the relevant gates from [week 20](week-20.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-141"></a>

## Day 141 — Online recurrence

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I29 — Module 29 — Attention IO Complexity and Online Softmax](../originals/inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md). Focus on online recurrence, using the task below to select the relevant subsection.

**External entry:** [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). See [resource verification notes](../RESOURCES.md).

**Do:** Derive the running maximum, denominator and weighted-value update for softmax.

**Done when:** A hand-worked split vector matches ordinary stable softmax.

**Career block:** Recall one coding or debugging pattern relevant to attention IO; write a tiny example. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-141.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-142"></a>

## Day 142 — Chunked implementation

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I29 — Module 29 — Attention IO Complexity and Online Softmax](../originals/inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md). Focus on chunked implementation, using the task below to select the relevant subsection.

**External entry:** [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). See [resource verification notes](../RESOURCES.md).

**Do:** Implement the recurrence in a readable CPU/PyTorch attention prototype.

**Done when:** Different chunk sizes give matching outputs within tolerance.

**Career block:** Take one next step on an active opportunity; if no search is active, save one role requirement related to attention IO. Use the [career reference](../originals/ai-career-acceleration/guides/job-search.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-142.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-143"></a>

## Day 143 — FlashAttention mechanism

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I30 — Module 30 — FlashAttention Evolution](../originals/inference-engineering-curriculum/modules/30-flashattention-evolution.md). Focus on flashattention mechanism, using the task below to select the relevant subsection.

**External entry:** [FlashAttention](https://arxiv.org/abs/2205.14135). See [resource verification notes](../RESOURCES.md).

**Do:** Read the original algorithm and map its IO argument to your prototype.

**Done when:** Separate algorithmic memory savings from GPU implementation claims.

**Career block:** Add one verified attention IO observation to an existing case study or resume evidence note. Use the [career reference](../originals/ai-career-acceleration/guides/profile-and-resume.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-143.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-144"></a>

## Day 144 — Mask and tail cases

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I10 — Module 10 — MHA, Masking, and Causal Attention](../originals/inference-engineering-curriculum/modules/10-mha-masking-and-causal-attention.md); [I29 — Module 29 — Attention IO Complexity and Online Softmax](../originals/inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md). Focus on mask and tail cases, using the task below to select the relevant subsection.

**External entry:** [“Attention Is All You Need”](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf). See [resource verification notes](../RESOURCES.md).

**Do:** Add causal masking, uneven lengths and documented behavior for fully masked rows.

**Done when:** Compare with the existing explicit attention oracle.

**Career block:** Maintain the single upstream work item or review a relevant test; avoid opening extra scope. Use the [career reference](../originals/ai-career-acceleration/guides/open-source-workflow.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-144.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-145"></a>

## Day 145 — Attention benchmark

**Time:** 135 min technical + 15 min career = 150 min.

**Read:** [I30 — Module 30 — FlashAttention Evolution](../originals/inference-engineering-curriculum/modules/30-flashattention-evolution.md); [P21 — Module 21 — Attention, KV Cache, and the PyTorch Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/21-attention-kv-cache-and-pytorch-bridge.md). Focus on attention benchmark, using the task below to select the relevant subsection.

**External entry:** [FlashAttention](https://arxiv.org/abs/2205.14135). See [resource verification notes](../RESOURCES.md).

**Do:** Compare oracle, prototype and available SDPA paths at feasible shapes.

**Done when:** Record peak memory, numerical error and the actual selected backend if observable.

**Career block:** Explain attention IO aloud in five minutes, then write the hardest follow-up question. Use the [career reference](../originals/ai-career-acceleration/guides/interviews.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-145.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-146"></a>

## Day 146 — Attention gate

**Time:** 165 min technical + 45 min career + 60 min weekly review = 270 min.

**Read:** [I29 — Module 29 — Attention IO Complexity and Online Softmax](../originals/inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md); [I30 — Module 30 — FlashAttention Evolution](../originals/inference-engineering-curriculum/modules/30-flashattention-evolution.md). Focus on attention gate, using the task below to select the relevant subsection.

**External entry:** [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867). See [resource verification notes](../RESOURCES.md).

**Do:** Explain the recurrence without notes and summarize when each path is useful.

**Done when:** No claim of matching a production FlashAttention kernel from a Python prototype.

**Career block:** Use this 45-minute block for a mock with debrief or the highest-value active application; use attention IO as evidence. Use the [career reference](../originals/ai-career-acceleration/guides/funnel-and-review.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-146.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Explain exact tiled attention without requiring a giant score matrix. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-147"></a>

## Day 147 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Explain exact tiled attention without requiring a giant score matrix.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
