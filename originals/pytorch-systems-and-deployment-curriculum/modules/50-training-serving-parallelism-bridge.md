# Module 50 — Training–Serving Parallelism Bridge

**Phase:** 7 — Distributed systems foundations  
**Time:** 12 focused hours  
**Prerequisites:** Modules 43–49 and inference curriculum Modules 29–34 and 45–47.

## Existing inference curriculum bridge

This is the explicit join between [29 — Attention I/O complexity and online softmax](../../inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md), [31 — PagedAttention and memory management](../../inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md), [33 — Speculative decoding](../../inference-engineering-curriculum/modules/33-speculative-decoding.md), [34 — Continuous batching, chunked prefill, and prefix cache](../../inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md), [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md), [46 — vLLM internals](../../inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md), and [47 — SGLang internals](../../inference-engineering-curriculum/modules/47-sglang-internals-and-contributions.md).

## Why this matters and outcomes

Training and serving share tensors, collectives, hardware, and sometimes code, but they optimize different objectives. Training chases steady-state tokens/s and model capacity under backward/optimizer memory. Serving balances time-to-first-token, inter-token latency, throughput under variable arrivals, KV-cache residency, and tail latency. Copying a training parallel configuration into a server is a common route to a correct but expensive or unstable system.

You will be able to translate a training partition into a serving hypothesis, distinguish weights from runtime state, choose metrics per phase, read a vLLM/SGLang/TensorRT-LLM design at the right abstraction, and design a fair cross-runtime experiment.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P50 — Training-serving parallelism bridge**.

1. [vLLM distributed serving](https://github.com/vllm-project/vllm) — learn the API or system contract.
2. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — study the underlying design, paper, or specification.
3. [PyTorch tensor parallel API](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

In training, data parallelism replicates weights and synchronizes gradients once per update; activation/optimizer state dominate different phases. In autoregressive serving, replicas often replicate weights to absorb requests, while tensor/pipeline/expert parallelism distributes one request/model across devices. The KV cache is request-specific persistent state, not an activation to discard after backward. Its allocation, reuse, paging, quantization, and prefix sharing can dominate memory and latency.

Prefill is GEMM-rich with long sequence dimensions; decode is often small-batch, memory- and launch-sensitive. A TP collective's impact therefore changes with batch and token position. Pipeline stages can improve model fit but add per-token hop latency. MoE exchange depends on active-token routing and batch composition. Continuous batching changes the actual collective shapes over time. A scheduler, cache manager, and runtime graph/cuda-graph strategy may matter more than the same model's offline throughput.

Use a workload vector: model/checkpoint/precision; prompt and generated-token distributions; arrival/concurrency; warm/cold cache state; sampling; request mix; server/runtime versions; hardware topology; and metrics such as TTFT, TPOT/ITL, end-to-end p50/p95/p99, tokens/s, cache hit rate, memory, and error rate. Preserve quality/correctness checks. On your 5060 Ti, benchmark one engine at a time with compatible versions and record only workloads that fit; cross-device/multi-GPU results remain a plan, not a comparison.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/serving/workload_spec.yaml` and `metric_definitions.md`. Include fixed prompt/output distributions, arrival schedule, warm-up, tokenization, sampling, stop conditions, and metric boundaries.
2. **Optimized:** write `parallelism_translation.md` for a compact decoder. Map DDP/FSDP/TP/PP/EP training concerns to serving replicas, TP/PP/EP, KV-cache ownership, and failure/recovery boundaries. Cite the source code/configuration you inspected.
3. **Integrated:** build `runtime_comparison_plan.md` for eager PyTorch, vLLM/SGLang, and optionally TensorRT-LLM. Define separate install environments, capability checks, comparable model formats, workload harness, numerical/output policy, and a “not comparable” exit condition.

## Three focused sessions

1. **State ledger:** classify every tensor in one decoder as immutable weights, request input, transient activation, persistent KV cache, scheduler metadata, or training-only state. Identify the owner and lifetime under each runtime.
2. **Phase-aware metrics:** run or simulate prefill/decode workload slices and calculate TTFT/TPOT/throughput separately. Explain why a single generated-tokens/s number hides a user-visible regression.
3. **Source walk:** trace one request through a selected runtime's scheduler, memory/cache allocator, worker/parallel layer, and sampler. Write down only observed behavior at a pinned revision, not assumed equivalence.

## Experiments and deliverables

Commit `workload_spec.yaml`, `metric_definitions.md`, `parallelism_translation.md`, `runtime_comparison_plan.md`, and `request_lifecycle.md`. Record model hash/precision, runtime commit/config, prompt/output/arrival distributions, tokenization/sampling, cache state, GPU memory policy, batch/concurrency, warm-up, all reported percentile definitions, quality checks, and explicit comparability exclusions.

## Common failure modes

- Calling a DDP replica strategy tensor parallelism because both use multiple GPUs.
- Benchmarking only long prefill and generalizing to interactive decode.
- Ignoring cache warmness, tokenization, or sampling when comparing runtimes.
- Treating FSDP checkpoint shards as an immediately serving-ready artifact.
- Reading a runtime's advertised feature as proof it is enabled for a particular model/version.

## Mastery gate

Write a defensible translation of a training plan into a serving plan, including every state owner and collective. Given two benchmark claims, identify the missing workload variables that make them incomparable. Produce one local server measurement with its full evidence bundle and no extrapolated multi-GPU conclusion.

## Primary sources, code paths, and OSS bridge

- [vLLM documentation](https://docs.vllm.ai/)
- [SGLang documentation](https://docs.sglang.ai/)
- [TensorRT-LLM documentation](https://docs.nvidia.com/tensorrt-llm/index.html)
- [PyTorch distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)

Read the exact scheduler/cache/parallelism files for the serving runtime and version under test, plus TorchTitan's training parallelism configuration. Useful contributions are workload-harness parity, a clearly bounded metric bug, or an evidence-rich issue; do not send broad “runtime X is slower” claims without a reproducer.

## Next

Module 51 begins deployment by turning eager PyTorch behavior into an explicit exported-program contract.
