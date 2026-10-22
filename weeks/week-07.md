# Week 07: One reference Transformer

Days 043–049 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Tested decoder block with explicit shape contracts.

**Prerequisite:** Pass the relevant gates from [week 06](week-06.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-043"></a>

## Day 043 — Decoder contract

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I06 — Module 06 — The Transformer Forward Pass](../originals/inference-engineering-curriculum/modules/06-transformer-forward-pass.md); [I07 — Module 07 — Tokenization, Embeddings, and Logits](../originals/inference-engineering-curriculum/modules/07-tokenization-embeddings-and-logits.md); [P20 — Module 20 — Transformer in PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/20-transformer-in-pytorch.md). Focus on decoder contract, using the task below to select the relevant subsection.

**External entry:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762). See [resource verification notes](../RESOURCES.md).

**Do:** Choose a tiny dense decoder configuration; write shape and parameter-count tables and implement embeddings plus output projection.

**Done when:** Check token bounds, vocabulary dimension and weight-tying assumptions.

**Career block:** Implement a basic rate limiter or bounded cache exercise. Use the [career reference](../originals/ai-career-acceleration/weeks/week-07.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-043.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-044"></a>

## Day 044 — Normalization and FFN

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I08 — Module 08 — Normalization, Residuals, and Stability](../originals/inference-engineering-curriculum/modules/08-normalization-residuals-and-stability.md); [I09 — Module 09 — FFNs, Activations, and SwiGLU](../originals/inference-engineering-curriculum/modules/09-ffn-activations-and-swiglu.md). Focus on normalization and ffn, using the task below to select the relevant subsection.

**External entry:** [Layer Normalization](https://arxiv.org/abs/1607.06450). See [resource verification notes](../RESOURCES.md).

**Do:** Implement RMSNorm and a SwiGLU-style FFN in readable PyTorch, keeping float64 checks for small inputs.

**Done when:** Verify epsilon placement, dimensions and extreme input behavior.

**Career block:** Take one high-fit application or interview next step. Use the [career reference](../originals/ai-career-acceleration/weeks/week-07.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-044.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-045"></a>

## Day 045 — Causal attention oracle

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I10 — Module 10 — MHA, Masking, and Causal Attention](../originals/inference-engineering-curriculum/modules/10-mha-masking-and-causal-attention.md). Focus on causal attention oracle, using the task below to select the relevant subsection.

**External entry:** [“Attention Is All You Need”](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf). See [resource verification notes](../RESOURCES.md).

**Do:** Write explicit QK transpose, mask, softmax and value aggregation before using an optimized API.

**Done when:** Changing future tokens cannot change earlier-token outputs.

**Career block:** Draft an explanation of cache bytes using the actual decoder plan. Use the [career reference](../originals/ai-career-acceleration/weeks/week-07.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-045.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-046"></a>

## Day 046 — Assemble the block

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P20 — Module 20 — Transformer in PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/20-transformer-in-pytorch.md); [I06 — Module 06 — The Transformer Forward Pass](../originals/inference-engineering-curriculum/modules/06-transformer-forward-pass.md). Focus on assemble the block, using the task below to select the relevant subsection.

**External entry:** [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). See [resource verification notes](../RESOURCES.md).

**Do:** Combine residuals, attention and FFN into a tiny decoder with named intermediates.

**Done when:** Each intermediate shape and dtype matches the written contract.

**Career block:** Trace one relevant runtime symbol for an interview story. Use the [career reference](../originals/ai-career-acceleration/weeks/week-07.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-046.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-047"></a>

## Day 047 — Reference comparison

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P20 — Module 20 — Transformer in PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/20-transformer-in-pytorch.md); [P21 — Module 21 — Attention, KV Cache, and the PyTorch Bridge](../originals/pytorch-systems-and-deployment-curriculum/modules/21-attention-kv-cache-and-pytorch-bridge.md). Focus on reference comparison, using the task below to select the relevant subsection.

**External entry:** [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). See [resource verification notes](../RESOURCES.md).

**Do:** Compare the attention oracle against PyTorch SDPA with the same mask and scale.

**Done when:** Cover batch one, odd sequence lengths and at least two head dimensions.

**Career block:** Rehearse a memory-pressure incident and distinguish evidence from guesses. Use the [career reference](../originals/ai-career-acceleration/weeks/week-07.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-047.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-048"></a>

## Day 048 — Decoder reconstruction

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [I06 — Module 06 — The Transformer Forward Pass](../originals/inference-engineering-curriculum/modules/06-transformer-forward-pass.md); [P20 — Module 20 — Transformer in PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/20-transformer-in-pytorch.md). Focus on decoder reconstruction, using the task below to select the relevant subsection.

**External entry:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762). See [resource verification notes](../RESOURCES.md).

**Do:** Rebuild attention from an empty file and run the decoder parity suite.

**Done when:** Explain the dominant parameters and memory allocations.

**Career block:** Review coding gaps, improve the existing portfolio page and act on one current opportunity. Use the [career reference](../originals/ai-career-acceleration/weeks/week-07.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-048.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Causal attention and feed-forward outputs agree with independent references. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-049"></a>

## Day 049 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Causal attention and feed-forward outputs agree with independent references.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
