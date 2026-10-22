# Module 47 — SGLang Internals and Contributions

**Phase:** 7 — Distributed runtimes and open source  
**Pacing:** Week 24, paired with Module 48; three 3-hour sessions and a scoped SGLang proposal  
**Prerequisites:** Modules 31–46; paged KV caches, continuous scheduling, prefix reuse, structured output, kernels, profiling, and distributed inference

## Why this matters

SGLang's serving runtime combines frontend managers, inter-process messaging, a memory-aware scheduler, RadixAttention prefix reuse, model execution, attention/kernel backends, sampling, structured generation, and distributed features. Its distinguishing prefix cache changes both request scheduling and KV ownership. A correct contribution requires tracing those state transitions—and recognizing that `sglang-kernel` is a separately released package whose integration may span multiple pull requests.

## Outcomes

You can:

- trace a request through tokenization, scheduler, model runner, cache, sampler, and detokenization;
- explain radix-tree prefix matching, block/KV ownership, eviction, and refcounts;
- locate batching, memory-pool, attention-backend, and structured-output boundaries;
- benchmark scheduler/cache changes with realistic hit rates and tail metrics;
- follow SGLang's current contribution, CI, kernel-release, and AI-assistance expectations.

## Mental model: managers around a stateful scheduler

A useful runtime slice is:

```text
HTTP/OpenAI request -> TokenizerManager
 -> scheduler process: waiting/running state + cache/memory pools
 -> ModelRunner -> model layers/attention/kernels -> sampling
 -> DetokenizerManager -> streamed response
```

Processes commonly communicate through queues/ZMQ; verify the exact topology and optional data-parallel components at the pinned commit. The scheduler admits tokens under memory and batch budgets, builds forward batches, and supplies cache locations. The model runner owns device-facing input preparation and forward execution. The detokenizer incrementally converts token IDs while preserving request order, stops, and streaming semantics.

RadixAttention stores reusable token prefixes in a radix tree. An edge can represent a token segment rather than one token. Lookup walks the longest matching prefix; partial edge matches require a split. Cache nodes reference KV locations, and active requests protect referenced nodes from eviction. If a prompt of length \(L\) matches \(m\) valid cached tokens, ideal prefill work becomes \(L-m\), but decode still attends over the full effective context. A useful workload metric is

\[
H_{\text{token}}=
\frac{\sum_r m_r}{\sum_r L_r},
\]

reported alongside lookup cost, eviction, TTFT, and memory. Request hit rate alone overstates reuse when hits are short.

The cache and scheduler interact: cache-aware ordering can improve reuse while delaying older requests; chunked prefill changes when nodes become reusable; memory pressure can evict a popular prefix just before use. Correct keys must cover token identity and state that changes KV values, including model/adapters and relevant multimodal inputs. Refcounts and locks must prevent recycling live KV slots.

## Implementation ladder

### 1. Reference

Implement `inference-lab/reference/radix_cache.py` with compressed token edges, longest-prefix lookup, split/insert, reference counts, and LRU leaf eviction. Use integer KV-slot identifiers and assert that no live slot is reused.

### 2. Optimized

Connect the cache to the Module 34 simulator in `inference-lab/engine/sglang_scheduler_sim.py`. Add cache-aware and FCFS policies, chunked prefill, memory admission, cancellation, and aging. Construct collision-free explicit keys before experimenting with hashed keys.

### 3. Integrated

At a pinned SGLang commit, trace a cold and warm shared-prefix workload using runtime metrics/logging. Compare simulator predictions with `bench_serving`; then isolate one discrepancy or missing invariant. If the candidate touches a kernel, separate the kernel-package change, release/version bump, and Python caller integration as the project guide requires.

## Three 3-hour sessions

1. **Vertical request:** map entrypoint, managers/messages, scheduler batches, model runner, sampling, detokenizer, and cancellation.
2. **Radix state:** implement tree operations, cache/memory-pool invariants, eviction, and shared-prefix workload experiments.
3. **Contributor loop:** choose a registered unit-test-sized issue, search history/owners, build a minimal test, benchmark if critical-path behavior changes, and write a local proposal.

## Experiments and deliverables

Commit to the learner lab:

- `reference/radix_cache.py`;
- `engine/sglang_scheduler_sim.py`;
- `tests/test_radix_cache_invariants.py`;
- `benchmarks/traces/shared-prefix-arrivals.jsonl`;
- `benchmarks/results/sglang-prefix-scheduler.json`;
- `notes/source_walks/sglang-request.md`;
- `notes/contributions/sglang-proposal.md`.

Record SGLang/kernel-package commits, model/tokenizer, processes/ranks, scheduler policy, token budget, memory pool, cache key/refcount transitions, hit tokens, evictions, request arrival trace, TTFT/ITL/output throughput/goodput, errors, warm-up, and cache-reset policy.

## Practical contribution ladder and current rules

1. Correct documentation while completing a verified source walk.
2. Add a fast registered unit test for real parser, manager, sampling, or cache logic.
3. Fix a bounded model/runtime bug with accuracy and regression coverage.
4. Change scheduler/RadixAttention only with invariants plus workload/tail evidence.
5. Add/optimize a kernel through the separate `sglang-kernel` release flow, then integrate its released version.

The official guide recommends documentation for new contributors, fast tests, minimizing device synchronization, avoiding duplicated critical-path work, running accuracy tests when outputs change, and using proper benchmark/profiling tools rather than GSM8K's incidental throughput. CI is resource-controlled and may require authorized labels; follow the current instructions instead of repeatedly triggering jobs.

For AI assistance, consult the live guide, PR template, repository instructions, and issue-specific requirements. An official SGLang unit-test contribution issue states that AI-assisted code is acceptable only when the contributor understands every line and rejects superficial auto-generated tests. Treat that as a floor: the human owns design, correctness, evidence, communication, and maintenance. Do not invent a broader policy where the repository has not published one.

## Common failure modes

- Treating RadixAttention as only a dictionary from prompts to outputs.
- Counting request hits instead of reused tokens.
- Reusing KV across state that changes model activations.
- Evicting a referenced node or recycling live memory-pool slots.
- Improving cache locality by starving old/unique prompts.
- Benchmarking only a warm, high-hit cache without a cold baseline.
- Using a noisy accuracy sanity script as a speed benchmark.
- Editing `sglang-kernel` and consuming it in one PR despite release separation.
- Writing mock-only tests that never exercise the invariant.

## Mastery gate

Trace one live request through current code areas and reproduce its scheduler/cache transitions in the learner simulator. Demonstrate radix edge splitting, longest-prefix reuse, safe refcounted eviction, cancellation, and bounded fairness. Produce a minimal real-logic test plus a scoped contribution proposal that identifies whether it belongs in Python runtime or the separate kernel package and lists the current project/AI rules checked.

## Canonical code areas and OSS bridge

Start under `python/sglang/srt/`: entrypoints, `managers/` (tokenizer, scheduler, detokenizer and batch construction), `mem_cache/` (radix cache and memory pools), `model_executor/`, attention layers/backends, sampling, and constrained/structured-output code. Inspect `sgl-kernel/` separately and mirror source paths in `test/registered/unit/`. [mini-sglang](https://github.com/sgl-project/mini-sglang) is an official educational bridge, not a substitute for the production trace.

## Primary and authoritative sources

- [SGLang source](https://github.com/sgl-project/sglang)
- [SGLang paper](https://arxiv.org/abs/2312.07104)
- [SGLang contribution guide](https://docs.sglang.ai/developer_guide/contribution_guide.html)
- [SGLang benchmark and profiling guide](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html)
- [Official unit-test contribution issue and AI quality rule](https://github.com/sgl-project/sglang/issues/20865)

## Next

Module 48 completes the core curriculum with llama.cpp/ggml: explicit GGUF metadata, compute graphs, quantized block types, heterogeneous backends, and unusually strict human-owned contribution rules.
