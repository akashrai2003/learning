# Module 01 — The Inference System Map

> **Phase:** 0 — Orientation  
> **Time:** Week 1; two 3-hour sessions plus a 2–4 hour independent lab  
> **Prerequisites:** Python fluency, basic neural-network vocabulary, and comfort reading process logs

## Why this matters

A language model is a function; an inference engine is a timed, stateful system around that function. Most disappointing deployments come from optimizing the wrong box: tuning a matrix kernel while requests wait in a queue, shrinking weights while KV cache dominates memory, or reporting model tokens/s while users experience poor time to first token (TTFT). This module gives every later optimization a location and a metric.

The cumulative project is a separate repository named `inference-lab/`, with `reference/`, `kernels/`, `engine/`, `benchmarks/`, `tests/`, and `notes/`. Do not build an engine yet. First make its boundaries observable.

## Outcomes

By the end, you can:

- trace a request from UTF-8 text to streamed text and name every state transition;
- distinguish prefill, decode, scheduling, model execution, sampling, and detokenization;
- separate control-plane work from GPU data-plane work;
- define TTFT, inter-token latency (ITL), end-to-end latency, throughput, and goodput;
- identify which subsystem owns weights, activations, KV state, queues, and caches.

## Mental model: two loops, two planes

The **request loop** is:

`admit → tokenize → queue → prefill → sample → decode repeatedly → detokenize → stream`.

The **engine loop** repeatedly selects runnable sequences, forms a token budget, launches a model step, updates per-request state, and returns sampled tokens. Its control plane manipulates request IDs, sequence lengths, block tables, priorities, and stop conditions. Its data plane moves tensors and executes kernels.

Prefill evaluates many prompt positions in parallel and creates KV state. Decode usually evaluates one new position per active sequence and reads all prior KV state. Consequently, prefill tends toward compute-rich matrix multiplication; low-batch decode repeatedly streams weights and cache and is often bandwidth- or launch-bound.

## Theory, metrics, and tensor contracts

Let batch size be \(B\), prompt length \(S\), hidden width \(D\), vocabulary size \(V\), layers \(L\), attention heads \(H\), and head width \(D_h=D/H\).

| Boundary | Logical shape |
|---|---|
| token IDs | `[B, S]` integer |
| token embeddings | `[B, S, D]` |
| per-layer Q/K/V | `[B, H, S, D_h]` |
| hidden output | `[B, S, D]` |
| logits | `[B, S, V]`, usually only the final position is sampled |
| decode token | `[B, 1]`; sequences may have different logical lengths |

For one request,

\[
\mathrm{TTFT}=t_{\text{queue}}+t_{\text{tokenize}}+t_{\text{prefill}}+
t_{\text{sample-first}}+t_{\text{stream}}.
\]

ITL is measured between visible token emissions, not by dividing total generation time by token count. Throughput is tokens completed per wall-clock second across all requests. **Goodput** counts only requests meeting an explicit service-level objective. A fast saturated server can have high throughput and zero goodput.

Keep separate clocks for enqueue, schedule, GPU submit, GPU complete, and stream. GPU work is asynchronous, so a CPU timestamp immediately after a launch does not mean computation is complete.

## Implementation ladder

1. In `notes/system-map.md`, draw the request and engine loops. Annotate ownership of weights, KV cache, tokenizer state, and output buffers.
2. Define a small `TraceEvent(request_id, phase, t_ns, token_count)` record in `engine/events.py`; do not couple it to a web framework.
3. In `reference/request_trace.py`, simulate two prompts with unequal lengths through admission, prefill, and five decode steps. Emit JSON Lines.
4. In `benchmarks/trace_report.py`, derive TTFT and per-token ITL from events. Reject missing or out-of-order phases instead of silently averaging them.
5. Write `tests/test_trace_report.py` with a hand-calculated timeline and one malformed trace.

This ladder establishes interfaces that later modules will fill with real tensor operations.

## Two 3-hour sessions

**Session A — map and measure**

- **0:00–0:35 — Whiteboard:** trace one chat request; mark process and device boundaries.
- **0:35–1:10 — Shape audit:** carry `[B,S,D]` through a decoder-only model and contrast prefill with one-token decode.
- **1:10–1:20 — Break.**
- **1:20–2:05 — Instrumentation lab:** implement `TraceEvent` and the deterministic request simulator.
- **2:05–2:40 — Metrics lab:** compute TTFT, ITL distribution, throughput, and SLO goodput.
- **2:40–3:00 — Read an OSS engine:** map the same concepts onto vLLM’s API server, EngineCore, and GPU model runner.

**Session B — simulate and explain**

- **0:00–0:45:** implement three scheduling policies over the same trace schema.
- **0:45–1:30:** generate concurrency and prompt-length sweeps with fixed workloads.
- **1:30–2:15:** plot TTFT, p95 ITL, throughput, and goodput; annotate tradeoffs.
- **2:15–3:00:** source-walk one real request and revise the system map with evidence.

## Experiments and deliverables

Submit the system map, trace schema, simulator, report script, and tests. Run three simulated policies: first-come-first-served, shortest-prompt-first, and a fixed token-budget batcher. Plot TTFT versus total throughput. Explain why the policy that maximizes throughput need not minimize tail TTFT.

Record assumptions beside every result: concurrency, prompt/output lengths, warm/cold state, and whether tokenization is included.

## Failure modes

- Calling checkpoint load time “inference latency.”
- Treating prefill tokens/s and decode tokens/s as interchangeable.
- Measuring asynchronous GPU launches with unsynchronized CPU timers.
- Averaging ITL and hiding p95 stalls from batch reshaping or cache misses.
- Omitting queueing, tokenization, or network time without labeling the metric.
- Assuming one process: production engines often separate API, scheduler/engine, and GPU workers.

## Mastery gate

Given an unfamiliar serving trace, produce a phase timeline, identify the critical path, and state one experiment that distinguishes queue, CPU, kernel-launch, compute, bandwidth, and KV-capacity bottlenecks. Pass only if all reported metrics have unambiguous start and stop events.

## Primary sources

- [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html)
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)

## Code paths to read

- vLLM: `vllm/entrypoints/openai/api_server.py`
- vLLM: `vllm/v1/engine/core.py`
- vLLM: `vllm/v1/worker/gpu_model_runner.py`
- Transformers: `src/transformers/generation/utils.py`

## OSS bridge

Your simulator has the same conceptual split as vLLM’s front end, EngineCore, scheduler, and model runner, but no framework-specific types. Preserve that boundary: reference math belongs under `reference/`; scheduling and state under `engine/`; timing harnesses under `benchmarks/`.

## Next

Module 02 makes the hardware/software stack reproducible so identical source code means something across machines and across time.
