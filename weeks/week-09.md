# Week 09: Checkpoint parity and a small C++ forward path

Days 057–063 · 17 hours total · 12 technical + 4 career + 1 review hour.

**Weekly output:** Checkpoint mapping plus native linear/norm comparison.

**Prerequisite:** Pass the relevant gates from [week 08](week-08.md); carry unresolved blockers forward explicitly.

The original module/week estimates do **not** add to these hours. Use only the sections and exercises named below. The first local resource is the main reading; other local links are supporting references. Read for 20–30 minutes, then implement. The external entry is an optional next click within that same reading block, not another assignment.

<a id="day-057"></a>

## Day 057 — Architecture manifest

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I17 — Module 17 — Modern Model Architecture Forensics](../originals/inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md); [I07 — Module 07 — Tokenization, Embeddings, and Logits](../originals/inference-engineering-curriculum/modules/07-tokenization-embeddings-and-logits.md). Focus on architecture manifest, using the task below to select the relevant subsection.

**External entry:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762). See [resource verification notes](../RESOURCES.md).

**Do:** Inspect a chosen public model config, tokenizer and tensor names; map them to the reference decoder.

**Done when:** Identify every unsupported architectural detail before loading weights.

**Career block:** Practice a coding problem selected from actual interview gaps. Use the [career reference](../originals/ai-career-acceleration/weeks/week-09.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-057.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-058"></a>

## Day 058 — Tokenizer boundaries

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [I07 — Module 07 — Tokenization, Embeddings, and Logits](../originals/inference-engineering-curriculum/modules/07-tokenization-embeddings-and-logits.md); [E08 — Week 08: Checkpoint parity, tokenization and cached generation](../originals/edge-inference-roadmap/weeks/week-08.md). Focus on tokenizer boundaries, using the task below to select the relevant subsection.

**External entry:** [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers/index). See [resource verification notes](../RESOURCES.md).

**Do:** Use the model's actual tokenizer assets and test special tokens, empty text and prompt formatting.

**Done when:** Token IDs and decoding behavior match the selected reference.

**Career block:** Screen one role and prepare the most relevant evidence link. Use the [career reference](../originals/ai-career-acceleration/weeks/week-09.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-058.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-059"></a>

## Day 059 — Checkpoint mapping

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [P16 — Module 16 — Serialization, Checkpoints, and Model Artifacts](../originals/pytorch-systems-and-deployment-curriculum/modules/16-serialization-checkpoints-and-model-artifacts.md); [E08 — Week 08: Checkpoint parity, tokenization and cached generation](../originals/edge-inference-roadmap/weeks/week-08.md). Focus on checkpoint mapping, using the task below to select the relevant subsection.

**External entry:** [Serialization note](https://docs.pytorch.org/docs/stable/notes/serialization.html). See [resource verification notes](../RESOURCES.md).

**Do:** Load only compatible weights and compare named intermediate tensors with the reference implementation.

**Done when:** Locate the first mismatch rather than judging only generated text.

**Career block:** Explain document structure or tokenizer correctness with a synthetic example. Use the [career reference](../originals/ai-career-acceleration/weeks/week-09.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-059.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-060"></a>

## Day 060 — Native linear and norm

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E09 — Week 09: C++ transformer kernels and checkpoint integration](../originals/edge-inference-roadmap/weeks/week-09.md); [E10 — Week 10: A complete educational C++ inference engine](../originals/edge-inference-roadmap/weeks/week-10.md). Focus on native linear and norm, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Port the already-tested linear and normalization operations to C++ using the existing reader and binding.

**Done when:** Compare a small matrix including tail dimensions against the Python oracle.

**Career block:** Reassess contribution scope; close an unsuitable local idea rather than forcing a PR. Use the [career reference](../originals/ai-career-acceleration/weeks/week-09.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-060.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-061"></a>

## Day 061 — Runtime source walk

**Time:** 120 min technical + 30 min career = 150 min.

**Read:** [E09 — Week 09: C++ transformer kernels and checkpoint integration](../originals/edge-inference-roadmap/weeks/week-09.md); [I48 — Module 48 — llama.cpp/ggml Internals and Contributions](../originals/inference-engineering-curriculum/modules/48-llamacpp-ggml-internals-and-contributions.md). Focus on runtime source walk, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Trace loading and one operator through llama.cpp/ggml at a pinned commit.

**Done when:** Record the entry point, tensor layout, dispatch and reference test.

**Career block:** Design a small document AI service with a quality contract. Use the [career reference](../originals/ai-career-acceleration/weeks/week-09.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-061.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

<a id="day-062"></a>

## Day 062 — Model report

**Time:** 120 min technical + 90 min career + 60 min weekly review = 270 min.

**Read:** [E10 — Week 10: A complete educational C++ inference engine](../originals/edge-inference-roadmap/weeks/week-10.md); [P20 — Module 20 — Transformer in PyTorch](../originals/pytorch-systems-and-deployment-curriculum/modules/20-transformer-in-pytorch.md). Focus on model report, using the task below to select the relevant subsection.

**External entry:** [C++ language curriculum](https://www.learncpp.com/). See [resource verification notes](../RESOURCES.md).

**Do:** Write a checkpoint-parity report and native-operation report; keep a complete C++ decoder as a later extension.

**Done when:** State exactly which operations and model paths are implemented.

**Career block:** Run a coding/design mock and handle the most promising application next step. Use the [career reference](../originals/ai-career-acceleration/weeks/week-09.md) only for this scoped action; do not also run its original daily schedule.

**Save:** implementation/test or source-walk evidence plus `notes/day-062.md` in your working lab. Record the actual minutes and evidence path in [daily progress](../trackers/daily-progress.csv).

**60-minute review:** 15 min closed-book recall, 20 min evaluate the weekly gate, 15 min career pipeline/review, 10 min choose the next session. Gate: Match one real or deliberately tiny checkpoint and one native operator. Use the [weekly review template](../templates/WEEKLY_REVIEW.md).

<a id="day-063"></a>

## Day 063 — Rest

No required reading, coding, career tasks or catch-up. If you move the long session here, protect another day off. Do not turn every missed task into weekend debt.

**Week gate:** Match one real or deliberately tiny checkpoint and one native operator.

[Weekly index](../WEEK_INDEX.md) · [Progress](../PROGRESS.md) · [Pacing rules](../STUDY_SYSTEM.md)
