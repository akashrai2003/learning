# Capstone 04 — Mini Paged and Continuously Batched Serving Engine

## Objective

Build a small engine that makes request scheduling and KV-cache ownership
explicit. It need not beat vLLM or SGLang. It must be correct, observable, and
useful for controlled experiments.

## Required behavior

- asynchronous request arrival;
- tokenization outside or explicitly inside measured latency;
- prefill and one-token decode phases;
- block-based KV allocation and logical-to-physical mapping;
- continuous batching;
- configurable token budget and sequence limit;
- cancellation and completion cleanup;
- chunked prefill;
- prefix-cache lookup/reuse;
- streaming output events;
- per-request and aggregate metrics.

Optional after correctness:

- speculative decoding;
- priority/deadline policy;
- swapping/preemption;
- multimodal encoder scheduling;
- disaggregated prefill/decode simulation.

## Architecture

Keep boundaries visible:

```text
frontend/admission
 → request state
 → scheduler
 → cache/block manager
 → batch/model runner
 → sampler
 → output processor
```

The model runner may initially use the reference decoder. Later, substitute a
small Hugging Face model or optimized operator without changing scheduler
invariants.

## Cache invariants

Test:

- no physical block has conflicting owners;
- reference counts match shared prefixes;
- freed blocks return exactly once;
- sequence logical positions map to correct tokens;
- partial/tail blocks preserve valid-token counts;
- cancellation and exceptions release all owned blocks;
- copy-on-write or sharing semantics are explicit;
- cached decode matches the non-paged reference.

Use randomized event sequences and an invariant checker.

## Scheduler experiments

Compare:

1. static batching;
2. decode-priority continuous batching;
3. token-budget scheduling;
4. chunked prefill;
5. prefix-aware scheduling/reuse.

For controlled synthetic arrivals, measure:

- TTFT, ITL/TPOT, E2E distributions;
- input/output/request throughput;
- goodput under declared SLOs;
- queue time by phase;
- cache utilization and internal fragmentation;
- prefix hit rate and saved prefill tokens;
- starvation/fairness;
- batch composition over time.

Plot the load curve through saturation.

## Event-driven simulator first

Before coupling to real GPU execution, implement a deterministic simulator whose
prefill/decode service times are functions of token counts and batch size. This
lets you test scheduling policies without noisy kernels.

Then calibrate the service-time model from local GPU measurements and finally
connect the actual model runner. Compare predicted versus measured behavior and
explain the residual.

## Prefix-cache requirements

Define:

- token/block hash input;
- model, adapter, position, modality, and cache-format identity;
- collision handling;
- eviction policy;
- partial-block treatment;
- invalidation;
- accounting for hit/miss/eviction.

String-prefix equality is insufficient if tokenization or model state differs.

## Speculative extension

If added, verify that accepted-token output exactly follows the target
distribution for the chosen algorithm. Record:

- draft cost;
- verification cost;
- mean accepted tokens;
- rollback/cache cost;
- end-to-end latency and throughput.

Do not report acceptance rate alone.

## Acceptance gate

Given a recorded sequence of arrivals, allocations, prefix hits, cancellations,
and completions, reproduce scheduler/cache state step by step. Demonstrate that
all invariants hold, then show a workload where chunked prefill improves decode
tail latency while changing TTFT or throughput in a predictable direction.

## Stretch work

- Rust or C++ control plane;
- multi-process model runner;
- CUDA graph batch-size table;
- LoRA-aware cache identity;
- multimodal processor cache;
- context-parallel cache mapping;
- prefill/decode disaggregation with transfer cost.

