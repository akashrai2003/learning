# Bridge to the Inference Engineering Curriculum

The two courses answer different questions about the same systems:

| Curriculum | Central question |
|---|---|
| [Inference engineering](../inference-engineering-curriculum/README.md) | How do model architecture, numerics, kernels, schedulers, and runtimes determine LLM/multimodal serving behavior? |
| This curriculum | How does PyTorch represent, transform, compile, distribute, export, and deploy those computations—and how do you extend it correctly? |

Study them as a braid. The inference course supplies the operator-level
motivation; this course supplies the framework and artifact-level mechanics.
Do not duplicate a derivation merely because it appears in two filenames.
Instead, implement the inference concept in PyTorch, then inspect what happens
when it enters the compiler, an extension, or a deployment runtime.

## Recommended pairing

| PyTorch phase / modules | Pair with inference modules | The joint outcome |
|---|---|---|
| 01–06: workflow and foundations | 01–05, 43–44 | A reproducible experiment and a defensible performance claim |
| 07–10: tensors, devices, autograd | 03–04, 36 | Predict layout/dtype/stream behavior before a kernel or model run |
| 11–16: state, data, training | 02, 06–08, 28 | Implement model state and serialize it without confusing framework state with model architecture |
| 17–21: functional transforms, Transformer, KV cache | 06–12, 29–34 | Build a compact PyTorch decoder, then compare reference and serving-oriented attention/cache designs |
| 22–24: AMP, tuning, TorchAO | 18–28 | Make numerical, quality, packing, and runtime claims separately |
| 25–28: domain and evaluation work | 15, 17, 44 | Treat vision/audio/multimodal input contracts and evaluation as systems constraints |
| 29–35: FX through Python custom ops | 39–42 | See how graph capture, operators, mutation/aliasing, and compiler legality constrain optimization |
| 36–42: C++/CUDA, build, graphs, regressions | 35–44 | Move a verified operator through CUDA, PyTorch integration, and reproducible build/debug loops |
| 43–50: distributed PyTorch | 13, 45 | Translate training sharding/collectives into serving parallelism and communication reasoning |
| 51–58: export/deployment/edge | 24, 28, 35, 37, 42 | Compare artifact contracts, runtime boundaries, numerical behavior, and hardware targets |
| 59–62: source/contribution | 46–48 | Make a narrowly scoped, testable contribution in the project that best fits your evidence |

## A concrete Transformer thread

Use one compact decoder model across both curricula. It should be small enough
to train/debug locally but shaped enough to reveal real concerns—causal masking,
RMSNorm or LayerNorm, RoPE, grouped-query attention, a KV cache, and a
SwiGLU-like MLP.

| Step | Learn in inference course | Implement/inspect here |
|---|---|---|
| Reference forward pass | 06–12 | Module 20: `nn.Module`, buffers, state dict, autograd, shape/property tests |
| Cached decoding | 12, 29–34 | Module 21: cache layout, inference mode, eager/compiled parity and timing |
| Precision and weight formats | 18–28 | Modules 22–24: AMP and TorchAO; then compare with the framework-independent quantization workbench |
| Compiler behavior | 39–42 | Modules 29–35: capture the model, explain graph breaks/guards, choose custom-op boundaries |
| Custom fast path | 37–41 | Modules 34–40: Triton or C++/CUDA implementation, registration, build and operator contract |
| Runtime integration | 46–48 | Modules 50–55: map model parallelism and artifact choices to vLLM/SGLang/TensorRT-LLM considerations |

The durable artifact is not a single fast benchmark. It is a repository where a
slow reference, tests, traces, numeric comparisons, compiled variants, exported
variants, and source-walk notes all explain one another.

## Module-specific bridges worth protecting

- **Module 24 ↔ Inference 18–28.** Use the inference course for quantization
  theory (calibration, GPTQ/AWQ, FP8/NVFP4, packing); use TorchAO to understand
  how a framework expresses a quantized tensor/workflow and lowers it into
  kernels. Never conclude a format is faster from bit-width alone.
- **Modules 29–35 ↔ Inference 39–42.** The inference course explains the
  compiler/kernels stack from the performance side. These modules explain why
  mutation, aliasing, fake/meta behavior, dynamic shapes, and operator schema
  determine whether an optimization survives `torch.compile` and export.
- **Module 40 ↔ Inference 43.** Your FlashInfer SM120 build is the anchor case:
  a native build has an environment contract, an ABI contract, a target-arch
  contract, and a host-resource budget. `MAX_JOBS=1` controls outer build
  parallelism; `FLASHINFER_NVCC_THREADS=1` bounds inner compilation work.
- **Modules 43–50 ↔ Inference 45–47.** Training and serving use different
  traffic patterns, but they share topology, collective, partitioning, and
  failure-recovery reasoning. Learn the mechanism once and then specify the
  workload-specific tradeoff.
- **Modules 51–58 ↔ Inference 28, 42, 44.** An exported/runtime artifact must
  retain semantics and be benchmarked under a declared shape distribution.
  Artifact portability is not equivalent to model correctness, and either is
  not equivalent to low latency.

## Work order when time is scarce

If a model or kernel question motivates you during the PyTorch core, take the
smallest needed detour into the inference curriculum, then return:

1. Need to implement attention? Read Inference 06, 10–12; do PyTorch 20–21.
2. Need to reason about low-bit behavior? Read Inference 18–21 first; do
   PyTorch 22–24.
3. Need a faster operator? Read Inference 35–44; do PyTorch 29–42.
4. Need a local serving system? Read Inference 29–34 and 46–48; do PyTorch
   43–58 as needed.

This keeps both courses coherent while preserving the habit that matters most:
every optimization starts from a concrete workload, an oracle, and a measured
constraint.
