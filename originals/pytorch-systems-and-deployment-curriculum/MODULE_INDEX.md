# Module Index

Each module is a standalone learning contract with a theory path, implementation
ladder, source walk, experiments, common failure modes, mastery gate, and a
specific bridge to the existing inference curriculum.

| # | Module | Primary output | Inference companion |
|---:|---|---|---|
| 01 | [Systems workflow for PyTorch](modules/01-systems-workflow-for-pytorch.md) | Evidence-based engineering loop | 01, 02, 44 |
| 02 | [Reproducible Python, CUDA, and packaging](modules/02-reproducible-python-cuda-and-packaging.md) | Version/build manifest | 02, 43 |
| 03 | [Python performance and data model](modules/03-python-performance-and-data-model.md) | Microbenchmark suite | 05 |
| 04 | [Tensor math and numerical contracts](modules/04-tensor-math-and-numerical-contracts.md) | Numerical oracle tests | 03, 04 |
| 05 | [Testing, debugging, and benchmark discipline](modules/05-testing-debugging-and-benchmark-discipline.md) | Test/benchmark harness | 05, 44 |
| 06 | [Framework landscape and interoperability](modules/06-framework-landscape-and-interoperability.md) | Artifact-selection memo | 28, 42 |
| 07 | [Tensor storage, views, strides, and layouts](modules/07-tensor-storage-views-strides-and-layouts.md) | Storage/stride visualizer | 03, 36 |
| 08 | [Shapes, indexing, broadcasting, and einsum](modules/08-shapes-indexing-broadcasting-and-einsum.md) | Shape property tests | 03, 06, 10 |
| 09 | [Devices, dtypes, memory, and streams](modules/09-devices-dtypes-memory-and-streams.md) | Async execution experiments | 04, 36 |
| 10 | [Autograd engine and gradient modes](modules/10-autograd-engine-and-gradient-modes.md) | Tiny autograd + grad tests | 06, 41 |
| 11 | [Modules, parameters, buffers, hooks, and state](modules/11-modules-parameters-buffers-hooks-and-state.md) | State-dict contract tests | 06, 28 |
| 12 | [torch.nn.functional and dispatch semantics](modules/12-torch-nn-functional-and-dispatch-semantics.md) | Dispatcher source walk | 41, 42 |
| 13 | [Data pipelines, datasets, and DataLoaders](modules/13-data-pipelines-datasets-and-dataloaders.md) | Input-pipeline profiler | 02, 44 |
| 14 | [Optimizers, schedulers, and training state](modules/14-optimizers-schedulers-and-training-state.md) | Resumable training state | 02, 04 |
| 15 | [Randomness, determinism, and reproducibility](modules/15-randomness-determinism-and-reproducibility.md) | Determinism matrix | 02, 44 |
| 16 | [Serialization, checkpoints, and model artifacts](modules/16-serialization-checkpoints-and-model-artifacts.md) | Compatibility dossier | 28, 43 |
| 17 | [torch.func, vmap, and functional transforms](modules/17-torch-func-vmap-and-functional-transforms.md) | Transform parity lab | 03, 41 |
| 18 | [PyTorch profiler, memory, and diagnostics](modules/18-pytorch-profiler-memory-and-diagnostics.md) | Chrome trace + memory analysis | 44 |
| 19 | [Minimal training engine](modules/19-minimal-training-engine.md) | Tested training loop | 06, 08 |
| 20 | [Transformer in PyTorch](modules/20-transformer-in-pytorch.md) | Compact decoder model | 06–12 |
| 21 | [Attention, KV cache, and PyTorch bridge](modules/21-attention-kv-cache-and-pytorch-bridge.md) | Cached decoding benchmark | 10, 12, 29–34 |
| 22 | [Automatic mixed precision and numerical stability](modules/22-automatic-mixed-precision-and-numerical-stability.md) | AMP stability report | 04, 25, 26 |
| 23 | [torchtune recipes and fine-tuning](modules/23-torchtune-recipes-and-finetuning.md) | Modified recipe/reproducer | 17, 18–28 |
| 24 | [TorchAO: quantization, sparsity, and low-bit](modules/24-torchao-quantization-sparsity-and-low-bit.md) | Quant/quality/perf ledger | 18–28 |
| 25 | [torchvision and vision model engineering](modules/25-torchvision-and-vision-model-engineering.md) | Vision pipeline profile | 15 |
| 26 | [torchaudio and streaming audio](modules/26-torchaudio-and-streaming-audio.md) | Streaming state test | 15 |
| 27 | [Multimodal models and processors](modules/27-multimodal-models-and-processors.md) | Multimodal contract test | 15, 17 |
| 28 | [Evaluation, quality, and experiment tracking](modules/28-evaluation-quality-and-experiment-tracking.md) | Reproducible evaluation dossier | 19, 28, 44 |
| 29 | [FX graphs and program transforms](modules/29-fx-graphs-and-program-transforms.md) | FX rewrite + tests | 39, 41, 42 |
| 30 | [torch.compile and TorchDynamo](modules/30-torch-compile-and-torchdynamo.md) | Compile benchmark baseline | 42, 44 |
| 31 | [Guards, recompilation, and dynamic shapes](modules/31-guards-recompilation-and-dynamic-shapes.md) | Guard trace dossier | 34, 42 |
| 32 | [AOTAutograd, functionalization, and autograd graphs](modules/32-aotautograd-functionalization-and-autograd-graphs.md) | Mutation/alias test suite | 41, 42 |
| 33 | [TorchInductor fusion, scheduling, and codegen](modules/33-torchinductor-fusion-scheduling-and-codegen.md) | Inductor trace analysis | 37–42 |
| 34 | [Triton with PyTorch](modules/34-triton-with-pytorch.md) | Parity + perf kernel | 37–42 |
| 35 | [torch.library and Python custom ops](modules/35-torch-library-and-python-custom-ops.md) | `opcheck`-clean custom op | 41, 42 |
| 36 | [C++ and CUDA custom operators](modules/36-cpp-cuda-custom-operators.md) | Packaged extension | 36–43 |
| 37 | [Dispatcher, ATen, and autograd in C++](modules/37-dispatcher-aten-and-autograd-in-cpp.md) | Dispatch-key/source map | 41, 42 |
| 38 | [native_functions, codegen, and operator lifecycle](modules/38-native-functions-codegen-and-operator-lifecycle.md) | YAML-to-test lifecycle map | 41–43 |
| 39 | [LibTorch and C++ frontend](modules/39-libtorch-and-cpp-frontend.md) | C++ inference exercise | 39–43 |
| 40 | [Build, JIT, Ninja, CMake, and ABI](modules/40-build-jit-ninja-cmake-and-abi.md) | Controlled build recovery | 36, 41, 43 |
| 41 | [CUDA Graphs, memory, and framework integration](modules/41-cuda-graphs-memory-and-framework-integration.md) | Graph-capture experiment | 35, 36, 44 |
| 42 | [Compiler debugging, minimization, and regression](modules/42-compiler-debugging-minimization-and-regression.md) | Minimal repro + bisect plan | 42–44 |
| 43 | [Process groups, collectives, and torch.distributed](modules/43-process-groups-collectives-and-torch-distributed.md) | Collective lab | 45 |
| 44 | [DistributedDataParallel](modules/44-distributed-data-parallel.md) | DDP scaling report | 45–47 |
| 45 | [FSDP and sharded training](modules/45-fsdp-and-sharded-training.md) | Memory-accounting lab | 45 |
| 46 | [DeviceMesh and DTensor](modules/46-devicemesh-and-dtensor.md) | Placement prototype | 45 |
| 47 | [Tensor, pipeline, context, and expert parallelism](modules/47-tensor-pipeline-context-and-expert-parallelism.md) | Parallelism design map | 13, 45 |
| 48 | [Distributed checkpointing, elasticity, and reliability](modules/48-distributed-checkpointing-elasticity-and-reliability.md) | Fault/recovery drill | 45 |
| 49 | [NCCL, topology, profiling, and communication performance](modules/49-nccl-topology-profiling-and-communication-performance.md) | Communication trace | 44–47 |
| 50 | [Training-serving parallelism bridge](modules/50-training-serving-parallelism-bridge.md) | Parallelism translation dossier | 45–47 |
| 51 | [torch.export and program capture](modules/51-torch-export-and-program-capture.md) | ExportedProgram contract | 17, 42 |
| 52 | [ONNX export and ONNX IR](modules/52-onnx-export-and-onnx-ir.md) | Validated ONNX graph | 28, 42 |
| 53 | [ONNX Runtime and execution providers](modules/53-onnx-runtime-and-execution-providers.md) | ORT parity/perf report | 28, 44 |
| 54 | [TensorRT and Torch-TensorRT](modules/54-tensorrt-and-torch-tensorrt.md) | Engine comparison | 35, 37, 44 |
| 55 | [TensorRT-LLM and runtime integration](modules/55-tensorrt-llm-and-runtime-integration.md) | Runtime mapping / deferral note | 34, 45–47 |
| 56 | [AOTInductor packaging and native deployment](modules/56-aotinductor-packaging-and-native-deployment.md) | AOT package dossier | 39–43 |
| 57 | [ExecuTorch architecture, EXIR, and backends](modules/57-executorch-architecture-exir-and-backends.md) | Partition/delegate map | 24, 28 |
| 58 | [ExecuTorch quantization, delegates, and edge](modules/58-executorch-quantization-delegates-and-edge.md) | Edge deployment case study | 18–28 |
| 59 | [PyTorch source tree and build](modules/59-pytorch-source-tree-and-build.md) | Local source-build plan | 41–43 |
| 60 | [PyTorch testing, CI, and triage](modules/60-pytorch-testing-ci-and-triage.md) | Triage/reproducer exercise | 41–43 |
| 61 | [PyTorch contribution engineering](modules/61-pytorch-contribution-engineering.md) | Upstream-quality patch plan | 46–48 |
| 62 | [PyTorch ecosystem capstone](modules/62-pytorch-ecosystem-capstone.md) | Contribution/project portfolio | 46–48 |

For implementation specifics and prerequisite order, use the individual module
instead of treating the table as a checklist.
