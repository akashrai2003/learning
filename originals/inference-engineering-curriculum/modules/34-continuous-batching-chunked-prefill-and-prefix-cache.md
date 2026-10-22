# Module 34 — Continuous Batching, Chunked Prefill, and Prefix Cache

**Phase:** 5 — Attention, decoding, and scheduling  
**Pacing:** Week 17, paired with Module 33; three 3-hour sessions plus the Phase 5 gate  
**Prerequisites:** Modules 12, 14, and 29–33; KV-cache blocks, prefill/decode costs, request metrics, and event-driven simulation

## Why this matters

An inference server rarely receives equal-length requests simultaneously. Static batches wait for every member; continuous batching admits and retires sequences at iteration boundaries. Chunked prefill divides long prompts so they can share a token budget with urgent decodes. Prefix caching reuses KV blocks when token prefixes and model state are identical. Together they determine GPU utilization, TTFT, TPOT, goodput, fragmentation, and fairness—often more than a single kernel improvement.

## Outcomes

You can:

- implement an iteration-level scheduler with sequence and token budgets;
- explain why prefill is usually compute-heavy while decode is often bandwidth-heavy;
- model chunk size as a TTFT/TPOT/fairness trade-off;
- design a content-addressed, block-aligned prefix cache with correct invalidation;
- measure goodput and starvation rather than only aggregate tokens/s.

## Mental model: schedule tokens, not requests

At scheduling iteration \(j\), choose work \(x_{r,j}\) for request \(r\) under

\[
\sum_r x_{r,j}\le T_{\max},\qquad
|\{r:x_{r,j}>0\}|\le S_{\max},
\]

plus available KV blocks. A decode sequence usually consumes one scheduled token; a prefill can consume a chunk. The batch is therefore ragged and changes every step.

Decode-priority chunked prefill first schedules ready decode tokens, then fills the remaining token budget with prefills. Smaller \(T_{\max}\) limits how long decodes wait behind prompt work and can improve inter-token latency. Larger budgets form more efficient prefill GEMMs and can improve TTFT or aggregate throughput. The optimum depends on arrival rate and SLOs, not a universal flag.

A prefix cache maps a block hash to a physical KV block. A safe key commits to the ordered token block and all state that changes its KV values: model/checkpoint identity, adapter, relevant multimodal inputs, position scheme, and cache dtype/configuration. Block \(i\)'s hash should chain from block \(i-1\), so identical suffix tokens under different prefixes cannot collide semantically. Only full blocks are broadly reusable; partial blocks require careful length and overwrite rules.

Caching saves prefill computation, not decode attention over the reused context. It also consumes blocks that could serve active requests. Eviction should account for recency/refcount and never recycle a block still referenced by a running sequence. In multi-tenant service, hash salt and isolation policy matter because hit timing can reveal shared prefixes.

Use four different measures:

- **TTFT:** arrival to first emitted token.
- **TPOT/ITL:** time between subsequent tokens.
- **Throughput:** completed prompt/output tokens per wall time.
- **Goodput:** requests or tokens meeting declared TTFT and TPOT SLOs.

A policy can raise throughput while lowering goodput or starving a long prompt.

## Implementation ladder

### 1. Reference

Build `inference-lab/reference/scheduler_sim.py`, a discrete-event simulator with arrivals, prompt/output lengths, prefill/decode cost functions, a paged KV allocator, and static versus continuous batching. Assert conservation of scheduled tokens and unique ownership/refcounts.

### 2. Optimized

Add chunked prefill, decode priority, configurable `max_num_batched_tokens`, limits on concurrent partial prefills, and aging. Implement `inference-lab/engine/prefix_cache.py` with chained hashes, collision verification, LRU eviction, reference counts, and block-aligned reuse. Preserve a no-cache oracle.

### 3. Integrated

Replay synthetic and recorded arrival traces through the simulator and a real engine. Sweep token budget, maximum sequences, chunk size, cache capacity, prefix popularity, and long/short-request mix. Compare cache-disabled, cold-cache, and warm-cache runs; reset state between independent trials.

## Three 3-hour sessions

1. **Continuous scheduler:** implement arrivals, iteration batches, memory admission, preemption, and TTFT/TPOT accounting.
2. **Chunks and fairness:** add partial prefills and aging; construct an adversarial stream of short requests around one long prompt.
3. **Prefix reuse:** implement chained block hashes and eviction, then validate simulator predictions against engine traces.

## Experiments and deliverables

Commit:

- `reference/scheduler_sim.py`;
- `engine/prefix_cache.py`;
- `tests/test_scheduler_invariants.py`;
- `benchmarks/traces/mixed-arrivals.jsonl`;
- `benchmarks/results/scheduler-policy-sweep.json`;
- `notes/source_walks/vllm-scheduler-prefix-cache.md`.

Report p50/p95/p99 TTFT and TPOT, throughput, goodput, queue time, cache-hit tokens, eviction/refcount events, preemptions, GPU token utilization, and KV fragmentation. Include a uniform offline batch, Poisson arrivals, a burst, shared-prefix chat traffic, and an adversarial long-prefill workload.

## Common failure modes

- Treating maximum sequences as the real compute budget instead of scheduled tokens.
- Measuring service time but omitting queue time.
- Allowing long prefills to monopolize every iteration.
- Giving decodes permanent priority without aging, so prompts starve.
- Counting token-string matches that are not block-aligned cache hits.
- Reusing across models, adapters, positions, or multimodal inputs that change KV.
- Evicting a referenced block or failing to decrement reference counts.
- Benchmarking a warm cache against a cold-cache baseline.
- Reporting total throughput without SLO distributions.

## Mastery gate

Given a timestamped mixed workload and fixed KV capacity, predict two scheduling iterations by hand, including block allocation. Then run the simulator to show where static batching, continuous batching, and chunked prefill differ. Demonstrate a valid prefix hit, a deliberately invalidated hit, safe eviction, and bounded wait for a long request. Defend one policy using goodput and tail metrics.

## Canonical code areas and OSS bridge

In vLLM, trace request admission through the V1 engine core into `vllm/v1/core/sched/`, scheduler outputs, KV-cache manager/block pool, worker model runner, and metrics. Read the [optimization guide](https://docs.vllm.ai/en/latest/configuration/optimization/) and the [automatic prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/) beside the [PagedAttention paper](https://arxiv.org/abs/2309.06180). In SGLang, compare the scheduler and RadixAttention/radix-cache paths under `python/sglang/srt/`.

Good entry contributions include an invariant test, clearer preemption/cache diagnostics, a trace-driven scheduler benchmark, or a reproducer for starvation or incorrect invalidation. Scheduler changes require workload distributions and tail-latency evidence, not a single throughput number.

## Primary and authoritative sources

- [vLLM optimization and chunked-prefill guide](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM automatic prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)
- [Sarathi-Serve paper and source](https://github.com/microsoft/sarathi-serve)
- [SGLang paper](https://arxiv.org/abs/2312.07104)
- [SGLang source](https://github.com/sgl-project/sglang)

## Next

Module 35 crosses the runtime–hardware boundary and maps these token batches onto the resources, instructions, and compatibility rules of the learner's Blackwell GPU.
