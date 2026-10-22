# Module 53 — ONNX Runtime and Execution Providers

**Phase:** 8 — Export and deployment  
**Time:** 15 focused hours  
**Prerequisites:** Module 52, profiling basics, and a validated ONNX artifact.

## Existing inference curriculum bridge

This module turns [28 — Quantized kernel and system integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [35 — GPU architecture and Blackwell](../../inference-engineering-curriculum/modules/35-gpu-architecture-and-blackwell.md), and [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md) into a portable-runtime investigation. It is especially useful for non-generative, vision, audio, or multimodal subgraphs where an ONNX boundary is realistic.

## Why this matters and outcomes

ONNX Runtime (ORT) optimizes an ONNX graph and assigns capable nodes/subgraphs to ordered Execution Providers (EPs). The chosen EP list, installed binaries, shape information, provider options, memory/stream interop, and fallback graph determine the result. A model “using CUDA EP” may still execute important portions on CPU and spend time copying tensors.

You will be able to establish eager-to-ORT parity, inspect provider assignment, select CPU/CUDA/TensorRT EPs deliberately, profile node-level behavior, and distinguish graph optimization, backend kernels, host-device transfer, and fallback in a benchmark report.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P53 — ONNX Runtime and execution providers**.

1. [ONNX Runtime docs](https://onnxruntime.ai/docs/) — learn the API or system contract.
2. [Execution provider documentation](https://onnxruntime.ai/docs/execution-providers/) — study the underlying design, paper, or specification.
3. [ONNX Runtime performance tuning](https://onnxruntime.ai/docs/performance/) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

An ORT `InferenceSession` loads a model, applies graph optimizations, and asks EPs for capabilities. With providers ordered `['CUDAExecutionProvider', 'CPUExecutionProvider']`, CUDA claims supported subgraphs and CPU executes fallback; the priority order is an execution policy, not proof of full GPU coverage. TensorRT EP may claim additional subgraphs but introduces engine/profile/cache constraints and may still fall back. Read the session/provider logs and profile trace rather than inferring assignment from installation success.

For an RTX 5060 Ti, install an ORT GPU package whose documented CUDA/cuDNN compatibility matches the chosen PyTorch/CUDA environment, or use a separate environment. Version collisions are common when a process imports both packages. Pin exact wheels, driver, CUDA runtime, and EP configuration. CUDA EP offers stream and allocator options; use external stream/device tensor interop only after baseline correctness, lifetime, and synchronization tests. A numpy round trip is a legitimate reference path but can dominate small inference latency.

Correct performance measurement separates session creation/engine compilation, warm-up, feed preparation/copies, inference execution, output handling, and end-to-end request latency. Profile ORT with its session profiler and use Nsight/PyTorch only where they answer a specific boundary question. Benchmarking a static single shape cannot support a claim about dynamic-serving traffic.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/ort/parity.py` that loads fixed inputs from Module 52 and compares PyTorch, ONNX reference tools, and ORT CPU outputs under named tolerances.
2. **Optimized:** add `ort_session_matrix.py` for CPU and, only when compatible, CUDA EP. It writes available providers, ordered providers/options, graph/session settings, warm-up results, node profile, outputs, and raw timing samples.
3. **Integrated:** produce `ort_execution_report.md` with profiler attribution, fallbacks, transfer boundaries, memory/stream policy, input-shape buckets, and a verdict limited to the tested artifact/hardware. Add TensorRT EP only as a separately version-pinned branch.

## Three focused sessions

1. **Session evidence:** build CPU first, enumerate providers, save optimized graph/profile information if supported, and show node-level ownership rather than assuming it.
2. **CUDA boundary:** in an isolated compatible environment, compare host numpy feeds with device/IO-binding flow only after proving output and stream ordering. Measure copies and inference separately.
3. **Fallback diagnosis:** induce or identify an unsupported node, inspect the EP partition/profile, decide whether to accept fallback, alter the graph, add an EP/kernel, or select another runtime.

## Experiments and deliverables

Commit `parity.py`, `ort_session_matrix.py`, `test_ort_shapes.py`, `provider_manifest.json`, `ort_profile.json`, and `ort_execution_report.md`. Include ORT/ONNX/PyTorch/driver/CUDA/cuDNN versions, package source, available/selected provider order/options, optimization level, shape buckets, input/output locations, first-run/warm timings, raw percentile samples, profile attribution, fallbacks, accuracy metrics, and environment isolation notes.

## Common failure modes

- Calling CUDA EP active because it appears in `get_available_providers()`.
- Timing session construction or data copies as model execution without labeling them.
- Loading incompatible CUDA/cuDNN libraries through a shared `LD_LIBRARY_PATH`.
- Trusting a single output equality case with no shape/precision boundary tests.
- Using TensorRT EP cache artifacts across unverified TensorRT/GPU/software combinations.

## Mastery gate

Provide an ORT report that identifies every selected provider and meaningful fallback region, with correctness and timing scope stated. Demonstrate one transfer or fallback bottleneck from profiler evidence and make a bounded remediation decision rather than a provider-label claim.

## Primary sources, code paths, and OSS bridge

- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)
- [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [ONNX Runtime profiling tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)

Read [ONNX Runtime source](https://github.com/microsoft/onnxruntime), especially `onnxruntime/core/providers/`, and provider-specific tests at the exact version. A productive contribution is a reduced provider-partition/parity issue or test; prove the graph and environment contract before attributing it to an EP.

## Next

Module 54 compares direct TensorRT and Torch-TensorRT paths for NVIDIA-targeted acceleration.
