# Module 33 — Speculative Decoding

**Phase:** 5 — Attention, decoding, and scheduling  
**Pacing:** Week 17, paired with Module 34; three 3-hour sessions plus the Phase 5 gate  
**Prerequisites:** Modules 12, 29–32; cached autoregressive decoding, probability distributions, sampling, KV-cache addressing, and serving metrics

## Why this matters

Ordinary decoding runs the full target model once per emitted token. Speculative decoding asks a cheaper proposer to draft several tokens, then verifies them together with the target. It can reduce inter-token latency when target decode is memory-bound, but it adds proposal work, verification tokens, cache bookkeeping, and rejection logic. A longer draft is not automatically faster, and an implementation that “usually matches greedy output” is not a correct sampler.

## Outcomes

You can:

- derive lossless rejection sampling for a target distribution \(p\) and draft \(q\);
- distinguish draft-model, n-gram/suffix, MTP, and EAGLE-style proposers;
- model speedup from acceptance, draft cost, verification cost, batch size, and memory;
- implement greedy and stochastic speculative decoding with cache rollback;
- measure accepted tokens per target step without hiding quality or tail-latency regressions.

## Mental model: propose, verify, commit

Starting from prefix \(x_{1:t}\), the proposer samples a chain \(y_1,\ldots,y_K\) from conditional distributions \(q_i\). One target pass evaluates the corresponding \(p_i\). For stochastic decoding, accept proposal \(y_i\) with

\[
a_i=\min\left(1,\frac{p_i(y_i)}{q_i(y_i)}\right).
\]

At the first rejection, sample a replacement from the normalized residual

\[
r_i(x)=\frac{\max(0,p_i(x)-q_i(x))}
{\sum_z\max(0,p_i(z)-q_i(z))}.
\]

If all \(K\) proposals are accepted, sample one bonus token from the target. This construction preserves the target distribution, up to implementation numerics. Greedy verification is simpler—accept the consecutive prefix equal to the target argmax—but is not the stochastic algorithm.

Let \(\alpha\) be conditional mean acceptance. Under a simplifying independent model, the expected accepted draft tokens are

\[
\mathbb{E}[A]=\sum_{i=1}^{K}\alpha^i
=\frac{\alpha(1-\alpha^K)}{1-\alpha}.
\]

Useful progress per cycle is roughly \(1+\mathbb{E}[A]\), including the replacement or bonus token. Speedup requires that proposal plus multi-token verification costs less per committed token than ordinary target decode. At higher live batch sizes, verification has effective batch \(B K\); it can become compute-bound and erase the gain. Acceptance alone is therefore not a performance result.

Proposers occupy a spectrum. N-gram and suffix lookup reuse token patterns with almost no model memory. A small independent draft is simple but duplicates weights and KV state. MTP predicts future tokens from target representations. EAGLE-family approaches draft features or tokens from target hidden states. Each changes compatibility, memory, training needs, and where cache state branches.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/speculative.py`. Implement a tiny categorical target and draft, seeded sampling, sequential acceptance, residual sampling, greedy mode, EOS handling, and \(K=0\). Prove distributional parity with frequency tests and exact small-vocabulary calculations.

### 2. Optimized

Add `inference-lab/engine/spec_decode.py` around the learner decoder. Batch verification of \(K\) positions, stage target and proposer KV writes, and commit only the accepted prefix. Handle rejection, bonus tokens, stop strings, maximum length, logprobs, and RNG advancement explicitly. Add n-gram proposal before introducing another neural model.

### 3. Integrated

Serve controlled traces at concurrency 1, 8, and 32. Sweep \(K\), prompt/output lengths, temperature, and proposer type. Record draft latency, verification latency, acceptance by position, committed tokens per target invocation, peak memory, TTFT, TPOT, and output tokens/s. Disable speculation as the paired baseline.

## Three 3-hour sessions

1. **Probability and oracle:** derive the rejection rule, implement the categorical reference, and test empirical target-distribution equality.
2. **Stateful decoder:** integrate staged KV-cache writes and rollback; fuzz EOS, rejection position, penalties, and seeded sampling.
3. **Systems experiment:** profile proposal/verification separately, sweep \(K\) and concurrency, and locate the crossover where speculation hurts.

## Experiments and deliverables

Commit:

- `reference/speculative.py`;
- `engine/spec_decode.py`;
- `tests/test_speculative_distribution.py`;
- `benchmarks/results/speculative-sweep.json`;
- `notes/source_walks/vllm-spec-decode.md`.

The result manifest must pin model/proposer revisions, tokenizer, sampling parameters, seeds, engine and kernel commits, GPU, batch/concurrency, warm-up, and whether CUDA graphs include every tested draft length. Plot acceptance by proposal position and TPOT against non-speculative decoding. Include at least one deliberately poor proposer.

## Common failure modes

- Comparing sampled token strings instead of output distributions.
- Forgetting the residual distribution after rejection.
- Reusing target logits computed after an uncommitted token.
- Advancing RNG differently between baseline and treatment without documenting it.
- Leaving rejected KV entries visible through sequence lengths or block tables.
- Reporting acceptance rate while proposal overhead makes latency worse.
- Tuning \(K\) at concurrency one and deploying it at a large continuous batch.
- Claiming “lossless” across different floating-point kernels as bitwise identity.

## Mastery gate

For a five-token vocabulary, derive one complete accept/reject example by hand, then show that your implementation converges to the target distribution. In the learner engine, demonstrate correct rollback at every rejection position and produce a workload matrix identifying when \(K=1,2,4,8\) wins or loses. Explain the result using measured proposal and verification cost rather than acceptance alone.

## Canonical code areas and OSS bridge

Trace vLLM from speculative configuration into its V1 model-runner speculator implementations, rejection sampler, scheduler output, and KV-cache updates; use the [current speculative-decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/) because supported methods evolve. Compare [vLLM Speculators](https://github.com/vllm-project/speculators), the original [speculative sampling paper](https://proceedings.mlr.press/v202/leviathan23a.html), [Medusa](https://proceedings.mlr.press/v235/cai24b.html), and the [EAGLE repository](https://github.com/SafeAILab/EAGLE).

A useful first contribution is a deterministic rejection-sampler test, a cache-rollback reproducer, a missing compatibility diagnostic, or a benchmark covering a neglected concurrency/draft-length regime. Do not submit speed claims without distribution checks and a no-speculation baseline.

## Primary and authoritative sources

- [Accelerating Large Language Model Decoding with Speculative Sampling](https://proceedings.mlr.press/v202/leviathan23a.html)
- [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/chen23m.html)
- [vLLM speculative decoding documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM source](https://github.com/vllm-project/vllm)
- [EAGLE source](https://github.com/SafeAILab/EAGLE)

## Next

Module 34 places speculative requests inside a live scheduler where token budgets, prefix reuse, long prefills, and fairness determine whether the local speedup survives.
