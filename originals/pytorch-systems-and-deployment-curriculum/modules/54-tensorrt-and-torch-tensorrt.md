# Module 54 — TensorRT and Torch-TensorRT

**Phase:** 8 — Export and deployment  
**Time:** 18 focused hours  
**Prerequisites:** Modules 51–53, Modules 35–41, and a version-pinned NVIDIA deployment environment.

## Existing inference curriculum bridge

This applies [28 — Quantized kernel and system integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [35 — GPU architecture and Blackwell](../../inference-engineering-curriculum/modules/35-gpu-architecture-and-blackwell.md), [37 — GEMM and tensor cores](../../inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md), [40 — CUTLASS/CuTe](../../inference-engineering-curriculum/modules/40-cutlass-and-cute.md), and [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md) to an engine-builder/runtime boundary.

## Why this matters and outcomes

TensorRT is an NVIDIA compiler/runtime that builds optimized engines for declared precision and shape profiles. Torch-TensorRT makes TensorRT subgraph lowering available from PyTorch's modern compiler/export world. Both are powerful target branches, not generic replacements for eager PyTorch, vLLM, or a hand-tuned custom kernel. Engine compatibility, builder time, dynamic profiles, graph partitioning, plugins, and version/GPU coupling are part of the product.

You will be able to choose direct TensorRT versus Torch-TensorRT, prepare controlled shape profiles, inspect supported/fallback subgraphs, validate precision/parity, benchmark build/cold/warm execution separately, and decide when an engine is the wrong artifact.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P54 — TensorRT and Torch-TensorRT**.

1. [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/) — learn the API or system contract.
2. [Torch-TensorRT docs](https://pytorch.org/TensorRT/) — study the underlying design, paper, or specification.
3. [Torch-TensorRT repository](https://github.com/pytorch/TensorRT) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Direct TensorRT commonly consumes ONNX or an explicitly constructed network, then builder tactics produce a serialized engine executed through contexts. For dynamic dimensions, `-1` is only a placeholder: each runtime dimension needs one or more min/opt/max optimization profiles, and the context must select a covering profile before enqueue. The opt shape is a performance hypothesis; a broad range may cost tactics/memory/performance. Engines and plugins have software/hardware compatibility constraints, so store builder/TensorRT/CUDA/driver/GPU/profile metadata with the artifact and validate after upgrades.

Torch-TensorRT's current Dynamo frontend is the default and accepts PyTorch modules, FX graphs, or `ExportedProgram`s through the modern capture path. `torch.compile` provides JIT flexibility; export/AOT paths are better for inspectable serialized artifacts. TorchScript and FX frontends remain legacy/maintenance paths; new work should use Dynamo/export unless a pinned compatibility requirement says otherwise. Torch-TensorRT may create a hybrid graph: TensorRT engines for supported regions plus PyTorch fallback regions. Measure and inspect that partition; do not call it a fully TensorRT engine.

On SM120, first check the specific TensorRT/Torch-TensorRT release support matrix and package compatibility. Use a dedicated environment and a small supported model to establish capability. If the stack does not support your driver/toolkit/GPU/model, stop at a documented build/export/compatibility dossier or test a CPU/portable branch; do not “fix” it by mixing arbitrary NVIDIA libraries into the vLLM environment.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/tensorrt/deployment_contract.md` with model hash, I/O dtypes/layouts, valid shape buckets, precision/accuracy target, plugin policy, and build/runtime compatibility matrix.
2. **Optimized:** use a compact supported vision/encoder or MLP model. Build a direct ONNX→TensorRT engine with explicit profiles and a Torch-TensorRT Dynamo/export variant. Save builder logs, subgraph partition information, and parity data.
3. **Integrated:** write `engine_dossier.md` and a fresh-process runner. Measure build, deserialize/context setup, first inference, warm inference, transfers, memory, and profile selection. If hardware support is unavailable, preserve the reproducible deferral evidence rather than invented throughput.

## Three focused sessions

1. **Engine contract:** derive min/opt/max shapes from a real workload distribution, build a single-profile baseline, and test rejection/selection at boundaries.
2. **Frontend comparison:** run modern Torch-TensorRT against direct TensorRT for the same declared artifact. Inspect hybrid fallback and document why it exists rather than optimizing blindly.
3. **Precision/plugins:** sweep FP32/FP16 and a supported quantized path with quality metrics. For an unsupported op, decide among graph rewrite, existing plugin, new current Plugin V3 work, or another runtime.

## Experiments and deliverables

Commit `deployment_contract.md`, engine-build scripts, `test_engine_parity.py`, `profile_matrix.csv`, builder logs, and `engine_dossier.md`. Include TensorRT/Torch-TensorRT/PyTorch/CUDA/driver versions, GPU/SM, model and ONNX hashes, profiles, tactics/cache policy, supported/fallback graph partitions, plugin libraries/hashes, build/cold/warm timing, input-transfer scope, output error/quality, memory, and a support/deferred statement.

## Common failure modes

- Declaring dynamic shapes without a covering optimization profile.
- Reporting build or cache warm time as runtime latency.
- Using an engine after unverified GPU/TensorRT/plugin compatibility changes.
- Interpreting a hybrid Torch-TensorRT graph as full engine coverage.
- Starting a new project on a legacy TorchScript/FX frontend without a compatibility reason.

## Mastery gate

Deliver one engine/artifact whose shape profiles, compatibility, partition, precision, and parity are auditable in a clean process. Explain a real unsupported region and choose a bounded next step. State precisely why the local SM120 setup can or cannot support the tested path.

## Primary sources, code paths, and OSS bridge

- [Torch-TensorRT explained](https://docs.pytorch.org/TensorRT/user_guide/torch_tensorrt_explained.html)
- [Torch-TensorRT legacy-frontends policy](https://docs.pytorch.org/TensorRT/legacy.html)
- [TensorRT inference library overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/index.html)
- [TensorRT dynamic shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)
- [TensorRT custom layers/plugins](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)

Read [Torch-TensorRT](https://github.com/pytorch/TensorRT), TensorRT samples, and the current support matrix at the pinned release. Contributions should include a minimal reproducible graph, exact profile/precision settings, parity tests, and a maintainer-approved scope; engine internals and plugin ABI are not a casual first patch.

## Next

Module 55 examines TensorRT-LLM as an LLM-specific runtime branch alongside—not instead of—vLLM and SGLang.
