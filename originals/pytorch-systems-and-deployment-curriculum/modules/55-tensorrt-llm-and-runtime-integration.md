# Module 55 — TensorRT-LLM and Runtime Integration

**Phase:** 8 — Export and deployment  
**Time:** 18 focused hours  
**Prerequisites:** Modules 50 and 54; inference curriculum Modules 29–34 and 45–47; an NVIDIA-supported environment target.

## Existing inference curriculum bridge

This is a deployment branch for [29 — Attention I/O complexity and online softmax](../../inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md), [30 — FlashAttention evolution](../../inference-engineering-curriculum/modules/30-flashattention-evolution.md), [31 — PagedAttention and memory management](../../inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md), [33 — Speculative decoding](../../inference-engineering-curriculum/modules/33-speculative-decoding.md), [34 — Continuous batching, chunked prefill, and prefix cache](../../inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md), [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md), [46 — vLLM internals](../../inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md), and [47 — SGLang internals](../../inference-engineering-curriculum/modules/47-sglang-internals-and-contributions.md).

## Why this matters and outcomes

TensorRT-LLM is NVIDIA's LLM-specific toolkit: PyTorch-facing model authoring/conversion and TensorRT engine construction coupled to LLM runtimes and serving APIs. It is neither a drop-in replacement for vLLM/SGLang nor merely “TensorRT with a chat endpoint.” It makes different choices about supported model families, engine build/deployment, kernels, cache management, batching, and operational integration. Learning it gives you another implementation vocabulary and a way to evaluate vendor-specialized stacks honestly.

You will be able to map a model and workload to TensorRT-LLM prerequisites, inspect its build/runtime boundary, compare it fairly with vLLM/SGLang, identify when its engine-oriented workflow is appropriate, and prepare a reproducible deferred branch when your local hardware/software combination is unsupported.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P55 — TensorRT-LLM and runtime integration**.

1. [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/) — learn the API or system contract.
2. [TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM) — study the underlying design, paper, or specification.
3. [TensorRT-LLM performance guide](https://nvidia.github.io/TensorRT-LLM/) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

TensorRT-LLM defines/model-converts an LLM, applies target-specific graph/kernels/quantization/parallelism choices, builds engine artifacts, then executes them through Python/C++ runtime components or a serving layer. The engine build is an optimization and compatibility event, not request-time work to hide in a latency chart. Model conversion, tokenizer/chat template, precision/quantization, max sequence/batch/token policies, KV-cache configuration, parallelism world, and runtime build must agree.

vLLM and SGLang are general-purpose serving runtimes with their own scheduler, cache allocator, execution backend ecosystem, and rapidly changing model support. TensorRT-LLM can offer NVIDIA-specific engine integration and runtime features; it does not supersede the study of vLLM's PagedAttention/scheduler or SGLang's radix/prefix and runtime architecture. Compare the same model behavior and workload, not product labels. An engine builder's offline profile specialization may win a constrained stable workload while a dynamic serving runtime may be easier to iterate or support a different model/cache/scheduler requirement. This is a hypothesis to test, not a universal hierarchy.

For RTX 5060 Ti/SM120, first read the exact TensorRT-LLM release's support matrix, installation requirements, model list, CUDA/TensorRT/Python compatibility, and relevant release notes. Use an isolated environment/container. If a model is not supported or the 16 GB memory budget cannot satisfy the intended engine/cache shape, produce a versioned conversion/build feasibility dossier and select a smaller supported model for mechanism practice. Do not contaminate the working vLLM NVFP4 environment with a broad vendor library path.

## Implementation ladder in `torch-lab/`

1. **Reference:** write `torch-lab/trtllm/feasibility_matrix.md`: model/checkpoint architecture, tokenizer, target precision, SM/GPU memory, software versions, requested max sequence/concurrency, supported conversion path, and a pass/fail/deferred reason.
2. **Optimized:** for one documented supported small model, capture conversion/build commands, engine manifest, build time/memory, and a deterministic tiny prompt set. Keep engine construction and server execution in separately timed scripts.
3. **Integrated:** create `runtime_mapping.md` comparing TensorRT-LLM, vLLM, and SGLang by artifact path, scheduler/cache ownership, parallelism, feature availability, observability, operating constraints, and workload hypotheses. Add a common workload harness only when model outputs/settings are genuinely comparable.

## Three focused sessions

1. **Compatibility source walk:** resolve every version/model/hardware prerequisite from primary documentation and the pinned source. Record what is verified, inferred, or unavailable before installing.
2. **Artifact lifecycle:** trace checkpoint conversion through engine build to runtime loading. Inspect logs/configs and make an explicit max-token/cache-memory budget; test correctness before performance.
3. **Cross-runtime experiment design:** define common prompts, sampling, tokenizer/template, warm-up, arrival/concurrency, TTFT/TPOT/throughput/error metrics, and an exclusion rule. Run only matched cells; leave unsupported cells blank with evidence.

## Experiments and deliverables

Commit `feasibility_matrix.md`, build/run scripts or an exact deferred recipe, `engine_manifest.json`, `runtime_mapping.md`, `workload_harness_spec.yaml`, and `trtllm_dossier.md`. Include runtime/model commits, image/wheel hashes, TensorRT/CUDA/driver/GPU/SM, conversion source, precision/quant config, engine shapes/parallelism/cache, build versus load versus request timing, tokenizer/template/sampling, output quality policy, memory, known limitations, and untested claims.

## Common failure modes

- Comparing an offline engine-build result to warmed server token throughput.
- Calling incompatible tokenizer/chat-template or sampling outputs a model-runtime regression.
- Assuming a TensorRT-LLM feature exists for every architecture/quantization/hardware release.
- Treating TensorRT-LLM as an alternative spelling of vLLM or SGLang.
- Mixing CUDA/TensorRT libraries from unrelated environments until import happens to work.

## Mastery gate

Produce a version-pinned feasibility decision for a target model on SM120, including a cache/memory budget and a clean defer path. For a supported small model, demonstrate engine/runtime correctness and a workload-scoped comparison plan that does not make a universal runtime ranking.

## Primary sources, code paths, and OSS bridge

- [TensorRT-LLM documentation](https://docs.nvidia.com/tensorrt-llm/index.html)
- [TensorRT-LLM GitHub source](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)
- [vLLM documentation](https://docs.vllm.ai/)
- [SGLang documentation](https://docs.sglang.ai/)

Read the pinned TensorRT-LLM `tensorrt_llm/`, build/conversion scripts, and executor/runtime code, then compare them with the pinned vLLM/SGLang code you already study. High-value contributions tend to be a supported-model conversion bug with a minimal checkpoint/config, a compatibility-document correction, or a focused benchmark-harness fix.

## Next

Module 56 returns to PyTorch-native deployment with AOTInductor packages for Python-free C++ execution.
