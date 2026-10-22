# Module 32 — Decoding, Sampling, and Logits Processing

**Phase:** 5 — Attention and decoding systems  
**Pacing:** Week 16, paired with Module 31; three 3-hour sessions plus a weekend sampler conformance suite  
**Prerequisites:** Modules 08, 12, and 31; logits, cached autoregressive decoding, reproducibility, and continuous request state

## Why this matters

The model produces logits, not text. Production decoding must apply per-request penalties and constraints, truncate a vocabulary distribution, sample reproducibly on GPU, return log probabilities, detect token/string stops, and preserve behavior as requests enter and leave a continuous batch. Processor order and edge semantics are observable API contracts.

## Outcomes

You can:

- derive greedy, temperature, top-k, top-p, and min-p sampling;
- implement sign-correct penalties and stable probability calculations;
- test seeded statistical correctness and forbidden-token invariants;
- reason about GPU vocabulary bandwidth and sorting-free approaches;
- integrate per-request RNG, stops, logprobs, and detokenization.

## Mental model: transform, filter, normalize, draw

At one decode step, logits are \(l\in\mathbb{R}^{B\times V}\). A clear reference pipeline is:

1. apply allowed-token masks, biases, penalties, and minimum-length EOS rules;
2. temperature-scale \(l'_v=l_v/T\) for \(T>0\);
3. filter by top-k/top-p/min-p according to defined order;
4. compute stable probabilities;
5. draw from the categorical distribution using the request’s RNG;
6. append token, update counts/grammar/stop state, and detokenize incrementally.

Order matters because filters do not commute. Pin runtime semantics in tests.

**Greedy:** choose \(\arg\max_v l_v\). Treat temperature zero as a separate mode; never divide by zero.

**Temperature:**

\[
p_T(v)=\frac{\exp(l_v/T)}{\sum_j\exp(l_j/T)}.
\]

Small \(T\) sharpens, large \(T\) flattens.

**Top-k:** retain the \(k\) largest logits. Handle \(k\ge V\), ties, and at least one retained token.

**Top-p / nucleus:** sort by probability, retain the smallest prefix \(C\) such that
\(\sum_{v\in C}p(v)\ge p_0\), then renormalize. The candidate count adapts to distribution sharpness.

**Min-p:** retain tokens satisfying
\[
p(v)\ge \alpha\max_jp(j).
\]

Because softmax is monotonic, this threshold can be implemented in logit space after accounting for temperature.

## Penalties and constraints

A common repetition penalty \(\rho>1\) transforms previously seen-token logits sign-wise:

\[
l'_v=
\begin{cases}
l_v/\rho,&l_v>0\\
\rho l_v,&l_v<0.
\end{cases}
\]

Presence/frequency penalties are often additive,

\[
l'_v=l_v-\lambda_p\mathbf{1}[c_v>0]-\lambda_fc_v,
\]

but API conventions vary. Prompt versus generated-token counts also vary. Test the engine you expose.

Bad-word, grammar, and structured-output processors set disallowed tokens to
\(-\infty\). If every token is masked, fail deterministically rather than sample NaN. EOS can be masked until `min_tokens`; `ignore_eos` is distinct from removing every stop condition.

Stop strings operate on decoded text and can cross token boundaries. Streaming servers may need to hold back a suffix long enough to decide whether it starts a stop string. Token stop IDs and text stops have different inclusion semantics.

## GPU and batching implications

Logits touch \(BV\) values every token. Sorting all \(V\) items can be expensive; production samplers fuse penalties, filtering, softmax reductions, multinomial draw, and top-logprob extraction or use selection/sorting-free algorithms. Returning all logprobs changes bandwidth and should be benchmarked separately.

Randomness must belong to a logical request, not its current batch row. Compaction, preemption, or a new neighbor must not change a fixed-seed sequence unless the documented contract permits it. Counter-based RNG keyed by request/sequence and draw offset is easier to reason about than one global state.

Beam search is not multinomial sampling: it retains high-scoring sequence hypotheses, accumulates log probabilities with length penalties, and forks/reorders KV-cache state. Keep its state and metrics separate.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/sampling.py` with explicit processor order, FP64 stable softmax, top-k/p/min-p masks, seeded categorical draw, penalties, EOS/min-token, and stop-token state.

### 2. Optimized

Vectorize per-request parameters and write or use a fused GPU sampler. Avoid full sorting when selection/prefix algorithms suffice. Benchmark sampling with and without top logprobs across \(B,V\).

### 3. Integrated

Add sampler state to the paged-engine simulator: request-scoped RNG/counts, multiple return sequences, cancellation, streaming text stop buffers, and batch compaction. Compare outputs and logprobs against a canonical implementation.

## Three 3-hour sessions

1. **Math/reference:** implement processors and filters; hand-check a tiny vocabulary including ties and all-masked cases.
2. **Statistical tests:** draw repeated samples, compare empirical frequencies to expected probabilities, and test deterministic request interleavings.
3. **Runtime/profile:** trace API parameters to sampler kernels, logprob extraction, stop handling, and detokenization; sweep \(B,V\) and filters.

The weekend suite defines a versioned sampling contract and records conformance results for the reference, framework, and serving engine.

## Experiments and deliverables

Commit:

- `reference/sampling.py`;
- `tests/test_sampling_properties.py`;
- `tests/test_sampling_interleavings.py`;
- `benchmarks/results/sampler-vocab-batch.json`;
- `notes/source_walks/logits-to-stream.md`.

Test greedy ties, temperature limits, top-k/p boundaries, min-p, sign-changing penalties, bad words, all-masked rows, min tokens/EOS, stop strings split across tokens, fixed seeds, batch reorder, and top-logprob accuracy.

## Common failure modes

- Applying processors in an undocumented order.
- Treating temperature zero as division.
- Computing top-p on logits rather than probabilities without a valid equivalent.
- Using one RNG stream for a changing batch.
- Sampling from an all-masked or NaN distribution.
- Applying repetition penalty with the same arithmetic to positive and negative logits.
- Emitting text before it is safe relative to a stop-string prefix.
- Benchmarking sampler latency without requested logprobs or detokenization.

## Mastery gate

Given a tiny logit vector and processor configuration, derive the exact retained distribution and seeded draw sequence. Then pass statistical, masking, stop, and continuous-batch interleaving tests while explaining the GPU kernel’s bandwidth and synchronization costs.

## Canonical code paths and OSS bridge

Read Hugging Face’s canonical [`generation/logits_process.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py) and generation loop, then vLLM’s [`sampling_params.py`](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py), V1 logits processors, top-k/top-p sampler, and output processor. Compare [FlashInfer sampling kernels](https://github.com/flashinfer-ai/flashinfer). Useful OSS contributions include edge-case conformance tests, request-RNG determinism tests, stop-buffer fixes, and filter-specific benchmarks.

## Primary sources

- [The Curious Case of Neural Text Degeneration / nucleus sampling](https://arxiv.org/abs/1904.09751)
- [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)
- [Hugging Face generation source](https://github.com/huggingface/transformers/tree/main/src/transformers/generation)
- [vLLM sampling source](https://github.com/vllm-project/vllm/tree/main/vllm/v1/sample)
- [FlashInfer sampling and decoding kernels](https://github.com/flashinfer-ai/flashinfer)

## Next

Module 33 builds speculative decoding on top of this exact sampling contract; Module 34 then schedules prefill, decode, and speculative work across continuously changing batches.
