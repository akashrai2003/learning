# PyTorch Systems & Deployment Source Atlas

This is the ordered reading and implementation companion for the 62 modules.
Each entry follows a practical sequence: **learn the contract → read the
derivation/specification → implement or profile it → inspect the owning source
tree**. Versions move quickly: pin the repository revision used by your local
environment before drawing implementation conclusions.

Use the links in a module’s **Guided study sources** section first; return here
for the fourth source and the broader path. Primary papers are preferred for
algorithms and official documentation/source trees for framework behavior.

## P01 — Systems workflow for PyTorch

1. [PyTorch contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) — repository workflow and review norms.
2. [PyTorch developer notes](https://docs.pytorch.org/docs/stable/community/contribution_guide.html) — build, test, and issue conventions.
3. [PyTorch benchmarks](https://github.com/pytorch/pytorch/tree/main/benchmarks) — establish a measurement baseline.
4. [PyTorch source tree](https://github.com/pytorch/pytorch) — trace one behavior from Python to C++.

## P02 — Reproducible Python, CUDA, and packaging

1. [PyTorch get-started matrix](https://pytorch.org/get-started/locally/) — make the compatibility tuple explicit.
2. [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) — distinguish driver and toolkit constraints.
3. [PyTorch C++ extension docs](https://docs.pytorch.org/docs/stable/cpp_extension.html) — compile-extension contracts.
4. [PEP 517](https://peps.python.org/pep-0517/) — understand isolated builds and build backends.

## P03 — Python performance and data model

1. [Python data model](https://docs.python.org/3/reference/datamodel.html) — special-method and object-cost semantics.
2. [Python profiling](https://docs.python.org/3/library/profile.html) — separate measurement from intuition.
3. [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — microbenchmark tensor work correctly.
4. [PyTorch benchmark utilities source](https://github.com/pytorch/pytorch/tree/main/torch/utils/benchmark) — inspect measurement implementation.

## P04 — Tensor math and numerical contracts

1. [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — framework-specific precision contracts.
2. [Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — the numerical foundation.
3. [PyTorch tensor documentation](https://docs.pytorch.org/docs/stable/tensors.html) — exercise dtypes and reductions.
4. [ATen native operators](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — inspect a numerically sensitive operator.

## P05 — Testing, debugging, and benchmark discipline

1. [PyTorch testing overview](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#testing) — test hierarchy and commands.
2. [pytest documentation](https://docs.pytest.org/) — fixtures, parametrization, and failure reduction.
3. [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — robust microbenchmarks.
4. [PyTorch test tree](https://github.com/pytorch/pytorch/tree/main/test) — study a regression test near your chosen subsystem.

## P06 — Framework landscape and interoperability

1. [torch.export](https://docs.pytorch.org/docs/stable/export.html) — PyTorch’s ahead-of-time program contract.
2. [ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html) — interchange boundaries and limitations.
3. [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/) — production NVIDIA runtime model.
4. [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/) — constrained-device deployment path.

## P07 — Tensor storage, views, strides, and layouts

1. [Tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html) — aliasing and non-contiguity.
2. [torch.as_strided](https://docs.pytorch.org/docs/stable/generated/torch.as_strided.html) — the sharpest stride tool.
3. [PyTorch tensor semantics](https://docs.pytorch.org/docs/stable/tensors.html) — verify storage/stride experiments.
4. [Tensor implementation](https://github.com/pytorch/pytorch/tree/main/c10/core) — follow storage and TensorImpl ownership.

## P08 — Shapes, indexing, broadcasting, and einsum

1. [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) — establish shape rules.
2. [torch.einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) — write tensor algebra precisely.
3. [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html) — foundation for advanced indexing.
4. [Tensor indexing implementation](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — locate indexing operators after reproducing them.

## P09 — Devices, dtypes, memory, and streams

1. [CUDA semantics in PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html) — streams, device state, and allocator APIs.
2. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — execution and synchronization model.
3. [Understanding CUDA memory usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — inspect allocator snapshots.
4. [PyTorch CUDA implementation](https://github.com/pytorch/pytorch/tree/main/c10/cuda) — source for stream and allocator boundaries.

## P10 — Autograd engine and gradient modes

1. [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — graph construction and saved tensors.
2. [Autograd API](https://docs.pytorch.org/docs/stable/autograd.html) — exercise modes and custom functions.
3. [Extending autograd](https://docs.pytorch.org/docs/stable/notes/extending.html) — implement a correct custom Function.
4. [Autograd engine source](https://github.com/pytorch/pytorch/tree/main/torch/csrc/autograd) — trace node scheduling.

## P11 — Modules, parameters, buffers, hooks, and state

1. [torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) — registration and state rules.
2. [Saving/loading models](https://docs.pytorch.org/docs/stable/notes/serialization.html) — state-dict contract.
3. [Module hooks](https://docs.pytorch.org/docs/stable/notes/modules.html) — instrument without changing model semantics.
4. [Module source](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py) — inspect registration logic.

## P12 — torch.nn.functional and dispatch semantics

1. [torch.nn.functional](https://docs.pytorch.org/docs/stable/nn.functional.html) — functional/module distinction.
2. [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — dispatch-key mental model.
3. [torch.library](https://docs.pytorch.org/docs/stable/library.html) — register and test a custom operator.
4. [Dispatcher source](https://github.com/pytorch/pytorch/tree/main/c10/core) — inspect key selection and operator handles.

## P13 — Data pipelines, datasets, and dataloaders

1. [Data loading tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html) — Dataset/DataLoader baseline.
2. [torch.utils.data](https://docs.pytorch.org/docs/stable/data.html) — workers, samplers, and collation contracts.
3. [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — quantify input starvation.
4. [Data utilities source](https://github.com/pytorch/pytorch/tree/main/torch/utils/data) — inspect worker lifecycle.

## P14 — Optimizers, schedulers, and training state

1. [torch.optim](https://docs.pytorch.org/docs/stable/optim.html) — optimizer/state-dict contracts.
2. [Adam paper](https://arxiv.org/abs/1412.6980) — derive adaptive moments.
3. [Learning-rate scheduler API](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — implement resume-safe scheduling.
4. [Optimizer source](https://github.com/pytorch/pytorch/tree/main/torch/optim) — read one optimizer’s foreach/fused paths.

## P15 — Randomness, determinism, and reproducibility

1. [Reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html) — identify every source of non-determinism.
2. [PyTorch random API](https://docs.pytorch.org/docs/stable/random.html) — control generators explicitly.
3. [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) — reason about asynchronous execution.
4. [Random source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen) — inspect generator implementation only after reproducing a case.

## P16 — Serialization, checkpoints, and model artifacts

1. [Serialization note](https://docs.pytorch.org/docs/stable/notes/serialization.html) — recommended state-dict workflow.
2. [torch.load security guidance](https://docs.pytorch.org/docs/stable/generated/torch.load.html) — understand `weights_only` and trust boundaries.
3. [Distributed checkpoint docs](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) — scale artifact handling safely.
4. [Serialization source](https://github.com/pytorch/pytorch/tree/main/torch/serialization.py) — trace archive behavior.

## P17 — torch.func, vmap, and functional transforms

1. [torch.func](https://docs.pytorch.org/docs/stable/func.html) — transforms and composability.
2. [UX limitations](https://docs.pytorch.org/docs/stable/func.ux_limitations.html) — avoid unsupported patterns.
3. [Per-sample gradients tutorial](https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html) — build a useful vmap experiment.
4. [functorch source](https://github.com/pytorch/pytorch/tree/main/torch/_functorch) — inspect transform lowering.

## P18 — PyTorch profiler, memory, and diagnostics

1. [Profiler API](https://docs.pytorch.org/docs/stable/profiler.html) — collect traceable evidence.
2. [CUDA memory snapshot](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — distinguish allocator state from GPU state.
3. [Nsight Systems](https://docs.nvidia.com/nsight-systems/) — inspect host/device overlap.
4. [Profiler source](https://github.com/pytorch/pytorch/tree/main/torch/profiler) — understand event attribution.

## P19 — Minimal training engine

1. [PyTorch training tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — establish a correct loop.
2. [Automatic mixed precision](https://docs.pytorch.org/docs/stable/amp.html) — add safe numerical scaling.
3. [PyTorch examples](https://github.com/pytorch/examples) — compare your engine with maintained examples.
4. [TorchRec training loop examples](https://github.com/pytorch/torchrec) — inspect a larger production-oriented codebase.

## P20 — Transformer in PyTorch

1. [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — PyTorch attention contract.
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — derive the architecture.
3. [PyTorch Transformer tutorial](https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html) — implement and train one.
4. [Attention implementation](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/transformers) — inspect kernel dispatch.

## P21 — Attention, KV cache, and PyTorch bridge

1. [SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — masks, causal mode, and GQA behavior.
2. [MQA paper](https://arxiv.org/abs/1911.02150) — KV-head reduction rationale.
3. [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180) — serving-time KV virtualization.
4. [PyTorch transformer source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/transformers) — compare framework and server paths.

## P22 — Automatic mixed precision and numerical stability

1. [AMP documentation](https://docs.pytorch.org/docs/stable/amp.html) — autocast and GradScaler semantics.
2. [Mixed Precision Training paper](https://arxiv.org/abs/1710.03740) — loss-scaling derivation.
3. [Numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — validate reductions and tolerances.
4. [AMP source](https://github.com/pytorch/pytorch/tree/main/torch/amp) — inspect eligibility and scaling behavior.

## P23 — Torchtune recipes and finetuning

1. [Torchtune documentation](https://pytorch.org/torchtune/stable/) — recipes and configuration model.
2. [Torchtune GitHub](https://github.com/pytorch/torchtune) — implement a tiny recipe locally.
3. [LoRA paper](https://arxiv.org/abs/2106.09685) — understand adapter parameterization.
4. [Torchtune recipes](https://github.com/pytorch/torchtune/tree/main/recipes) — trace a recipe into its components.

## P24 — TorchAO quantization, sparsity, and low bit

1. [TorchAO documentation](https://docs.pytorch.org/ao/stable/) — current PyTorch-native low-bit APIs.
2. [TorchAO repository](https://github.com/pytorch/ao) — pin the revision and run an example.
3. [GPTQ paper](https://arxiv.org/abs/2210.17323) — second-order weight quantization.
4. [TorchAO prototype code](https://github.com/pytorch/ao/tree/main/torchao) — inspect quantizer and kernel boundaries.

## P25 — Torchvision and vision model engineering

1. [Torchvision docs](https://docs.pytorch.org/vision/stable/) — models, transforms, and weights API.
2. [Torchvision repository](https://github.com/pytorch/vision) — run a reference training/eval path.
3. [ResNet paper](https://arxiv.org/abs/1512.03385) — architecture baseline.
4. [Torchvision models source](https://github.com/pytorch/vision/tree/main/torchvision/models) — compare paper structure with code.

## P26 — Torchaudio and streaming audio

1. [TorchAudio documentation](https://docs.pytorch.org/audio/stable/) — APIs and current maintenance status.
2. [TorchCodec repository](https://github.com/pytorch/torchcodec) — learn the migration target for media decoding.
3. [TorchAudio repository](https://github.com/pytorch/audio) — reproduce a streaming path.
4. [TorchAudio source](https://github.com/pytorch/audio/tree/main/src/torchaudio) — inspect transforms and I/O boundaries.

## P27 — Multimodal models and processors

1. [PyTorch multimodal examples](https://pytorch.org/blog/) — locate current reference projects and APIs.
2. [CLIP paper](https://arxiv.org/abs/2103.00020) — contrastive multimodal objective.
3. [Torchvision transforms v2](https://docs.pytorch.org/vision/stable/transforms.html) — implement preprocessing contracts.
4. [TorchMultimodal repository](https://github.com/facebookresearch/multimodal) — inspect component composition.

## P28 — Evaluation, quality, and experiment tracking

1. [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — performance measurement.
2. [EleutherAI evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) — reproducible LLM quality evaluation.
3. [MLflow tracking docs](https://mlflow.org/docs/latest/ml/tracking/) — artifact and metric provenance.
4. [PyTorch benchmark tree](https://github.com/pytorch/pytorch/tree/main/benchmarks) — study workload definition and result reporting.

## P29 — FX graphs and program transforms

1. [FX documentation](https://docs.pytorch.org/docs/stable/fx.html) — symbolic tracing and graph transformations.
2. [FX limitations](https://docs.pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing) — learn where tracing breaks.
3. [FX graph mode quantization](https://docs.pytorch.org/docs/stable/quantization.html) — apply a graph transformation.
4. [FX source](https://github.com/pytorch/pytorch/tree/main/torch/fx) — trace a transform implementation.

## P30 — torch.compile and TorchDynamo

1. [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) — public contract and knobs.
2. [TorchDynamo deep dive](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html) — bytecode capture model.
3. [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — compiler-stack design.
4. [Dynamo source](https://github.com/pytorch/pytorch/tree/main/torch/_dynamo) — inspect graph breaks and guards.

## P31 — Guards, recompilation, and dynamic shapes

1. [Dynamic shapes manual](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html) — symbolic shape contracts.
2. [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) — diagnose recompiles.
3. [Dynamo guards source](https://github.com/pytorch/pytorch/tree/main/torch/_dynamo) — inspect guard construction.
4. [ShapeEnv source](https://github.com/pytorch/pytorch/tree/main/torch/fx/experimental) — follow symbolic reasoning.

## P32 — AOTAutograd, functionalization, and autograd graphs

1. [AOTAutograd source](https://github.com/pytorch/pytorch/tree/main/torch/_functorch) — start from the implementation owned by PyTorch.
2. [Functionalization source](https://github.com/pytorch/pytorch/tree/main/torch/_functorch) — study mutation removal.
3. [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — compiler-level context.
4. [Autograd source](https://github.com/pytorch/pytorch/tree/main/torch/csrc/autograd) — connect capture to backward execution.

## P33 — TorchInductor fusion, scheduling, and codegen

1. [PyTorch compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) — compiler configuration and generated code.
2. [Inductor CPU backend paper](https://arxiv.org/abs/2301.13448) — fusion/codegen design background.
3. [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — end-to-end architecture.
4. [Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — trace scheduler and code cache.

## P34 — Triton with PyTorch

1. [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) — write kernels in increasing complexity.
2. [Triton language reference](https://triton-lang.org/main/python-api/triton.language.html) — understand layouts and primitives.
3. [Triton repository](https://github.com/triton-lang/triton) — compile and inspect generated kernels.
4. [PyTorch Inductor Triton code](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — see production integration.

## P35 — torch.library and Python custom ops

1. [torch.library](https://docs.pytorch.org/docs/stable/library.html) — operator registration surface.
2. [Custom Python operators tutorial](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html) — implement dispatch/autograd correctly.
3. [Custom ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — choose the appropriate extension mechanism.
4. [torch.library source](https://github.com/pytorch/pytorch/tree/main/torch/library.py) — inspect schema registration behavior.

## P36 — C++/CUDA custom operators

1. [Custom C++ and CUDA operators tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — full implementation workflow.
2. [C++ extension docs](https://docs.pytorch.org/docs/stable/cpp_extension.html) — build controls and ABI boundaries.
3. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — write safe kernels.
4. [PyTorch extension examples](https://github.com/pytorch/extension-cpp) — inspect a maintained minimal project.

## P37 — Dispatcher, ATen, and autograd in C++

1. [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — dispatch-key foundations.
2. [C++ API docs](https://docs.pytorch.org/cppdocs/) — public native surface.
3. [ATen native source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — map schema to kernel.
4. [Autograd C++ source](https://github.com/pytorch/pytorch/tree/main/torch/csrc/autograd) — follow autograd wrapping.

## P38 — Native functions, codegen, and operator lifecycle

1. [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) — the operator declaration source of truth.
2. [PyTorch codegen source](https://github.com/pytorch/pytorch/tree/main/torchgen) — learn generated artifacts.
3. [Custom ops tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — understand lifecycle requirements.
4. [ATen templates](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen) — trace one declaration into generated code.

## P39 — LibTorch and C++ frontend

1. [PyTorch C++ frontend tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html) — create/train modules in C++.
2. [C++ API docs](https://docs.pytorch.org/cppdocs/) — API reference.
3. [TorchScript deprecation note](https://docs.pytorch.org/docs/stable/jit.html) — know legacy constraints before using them.
4. [C++ API source](https://github.com/pytorch/pytorch/tree/main/torch/csrc/api) — inspect frontend mapping.

## P40 — Build, JIT, Ninja, CMake, and ABI

1. [PyTorch from-source build](https://github.com/pytorch/pytorch#from-source) — reproduce a clean native build.
2. [C++ extension docs](https://docs.pytorch.org/docs/stable/cpp_extension.html) — extension JIT/build behavior.
3. [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/) — target and dependency model.
4. [Ninja manual](https://ninja-build.org/manual.html) — read the generated build graph and control parallelism.

## P41 — CUDA graphs, memory, and framework integration

1. [PyTorch CUDA graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#constraints) — capture constraints in PyTorch.
2. [CUDA Graphs guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) — runtime semantics.
3. [CUDA memory snapshot](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — establish stable allocation evidence.
4. [CUDAGraph source](https://github.com/pytorch/pytorch/tree/main/torch/cuda) — inspect framework capture plumbing.

## P42 — Compiler debugging, minimization, and regression

1. [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) — first-line diagnostic workflow.
2. [TorchDynamo FAQ](https://docs.pytorch.org/docs/stable/torch.compiler_faq.html) — common graph-break/recompile causes.
3. [Dynamo repro tooling source](https://github.com/pytorch/pytorch/tree/main/torch/_dynamo) — create minimized cases.
4. [PyTorch issue template](https://github.com/pytorch/pytorch/issues/new/choose) — write an actionable compiler report.

## P43 — Process groups, collectives, and torch.distributed

1. [torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html) — process-group and collective API contracts.
2. [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — backend algorithm/topology behavior.
3. [Distributed communication package source](https://github.com/pytorch/pytorch/tree/main/torch/distributed) — inspect process-group abstraction.
4. [c10d source](https://github.com/pytorch/pytorch/tree/main/torch/csrc/distributed/c10d) — follow collective implementation.

## P44 — DistributedDataParallel

1. [DDP API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — reducer and usage constraints.
2. [DDP design note](https://docs.pytorch.org/docs/stable/notes/ddp.html) — bucket/all-reduce mechanics.
3. [PyTorch distributed tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) — implement and debug a baseline.
4. [DDP source](https://github.com/pytorch/pytorch/tree/main/torch/nn/parallel) — inspect reducer setup.

## P45 — FSDP and sharded training

1. [FSDP API](https://docs.pytorch.org/docs/stable/fsdp.html) — sharding and state-dict contracts.
2. [FSDP advanced tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html) — tune a concrete workload.
3. [ZeRO paper](https://arxiv.org/abs/1910.02054) — memory-sharding rationale.
4. [FSDP source](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp) — inspect unshard/reshard sequencing.

## P46 — DeviceMesh and DTensor

1. [DTensor and DeviceMesh API](https://docs.pytorch.org/docs/stable/distributed.tensor.html) — logical topology abstraction.
2. [DTensor API](https://docs.pytorch.org/docs/stable/distributed.tensor.html) — placements and redistribution.
3. [Tensor parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) — implement a mesh-backed transformation.
4. [DTensor source](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor) — trace placement propagation.

## P47 — Tensor, pipeline, context, and expert parallelism

1. [PyTorch tensor parallel API](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) — native tensor-parallel surface.
2. [Pipeline parallel API](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) — scheduling and stage contracts.
3. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — study production parallelism composition.
4. [PyTorch distributed source](https://github.com/pytorch/pytorch/tree/main/torch/distributed) — compare APIs with Megatron implementation choices.

## P48 — Distributed checkpointing, elasticity, and reliability

1. [Distributed checkpointing](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) — checkpoint planner/storage contracts.
2. [TorchElastic](https://docs.pytorch.org/docs/stable/elastic/run.html) — failure and membership behavior.
3. [FSDP state-dict docs](https://docs.pytorch.org/docs/stable/fsdp.html) — capture sharded training state correctly.
4. [Distributed checkpoint source](https://github.com/pytorch/pytorch/tree/main/torch/distributed/checkpoint) — inspect save/load planner implementation.

## P49 — NCCL topology, profiling, and communication performance

1. [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — topology and environment controls.
2. [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) — diagnose transport problems safely.
3. [Nsight Systems](https://docs.nvidia.com/nsight-systems/) — measure collective overlap and stalls.
4. [NCCL source](https://github.com/NVIDIA/nccl) — inspect algorithm/protocol selection after profiling.

## P50 — Training-serving parallelism bridge

1. [vLLM source](https://github.com/vllm-project/vllm) — trace serving topology concepts in the current release.
2. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — training parallelism reference.
3. [PyTorch tensor parallel API](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) — framework bridge.
4. [vLLM source](https://github.com/vllm-project/vllm) — trace a serving parallel worker layout.

## P51 — torch.export and program capture

1. [torch.export](https://docs.pytorch.org/docs/stable/export.html) — capture contract and soundness model.
2. [Export programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/programming_model.html) — constraints and graph semantics.
3. [Export tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html) — capture and validate a model.
4. [Export source](https://github.com/pytorch/pytorch/tree/main/torch/export) — inspect verifier and serialization flow.

## P52 — ONNX export and ONNX IR

1. [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html) — exporter route and diagnostics.
2. [ONNX IR specification](https://onnx.ai/onnx/intro/) — graph, types, and operator-set semantics.
3. [ONNX exporter tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html) — validate with a small model.
4. [PyTorch ONNX source](https://github.com/pytorch/pytorch/tree/main/torch/onnx) — trace lowering decisions.

## P53 — ONNX Runtime and execution providers

1. [ONNX Runtime docs](https://onnxruntime.ai/docs/) — runtime architecture.
2. [Execution provider documentation](https://onnxruntime.ai/docs/execution-providers/) — partitioning and hardware backends.
3. [ONNX Runtime performance tuning](https://onnxruntime.ai/docs/performance/) — benchmark a provider fairly.
4. [ONNX Runtime source](https://github.com/microsoft/onnxruntime) — inspect graph partitioning and EP integration.

## P54 — TensorRT and Torch-TensorRT

1. [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/) — builder, engine, and runtime semantics.
2. [Torch-TensorRT docs](https://pytorch.org/TensorRT/) — PyTorch integration paths.
3. [Torch-TensorRT repository](https://github.com/pytorch/TensorRT) — compile and benchmark a model.
4. [TensorRT samples](https://github.com/NVIDIA/TensorRT/tree/main/samples) — inspect engine construction patterns.

## P55 — TensorRT-LLM and runtime integration

1. [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/) — LLM build/runtime workflow.
2. [TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM) — run a supported model path.
3. [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/) — use the current performance guide for your pinned release.
4. [TensorRT-LLM runtime source](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm) — inspect scheduling and kernels.

## P56 — AOTInductor packaging and native deployment

1. [AOTInductor documentation](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) — artifact and runtime contract.
2. [torch.export](https://docs.pytorch.org/docs/stable/export.html) — prerequisite capture semantics.
3. [PyTorch C++ custom-op tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — understand native extension boundaries.
4. [Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — inspect package/codegen behavior.

## P57 — ExecuTorch architecture, EXIR, and backends

1. [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/) — deployment lifecycle and architecture.
2. [ExecuTorch runtime overview](https://docs.pytorch.org/executorch/stable/runtime-overview.html) — portable runtime contracts.
3. [ExecuTorch repository](https://github.com/pytorch/executorch) — build a minimal backend target.
4. [ExecuTorch runtime source](https://github.com/pytorch/executorch/tree/main/runtime) — inspect executor/backend interface.

## P58 — ExecuTorch quantization, delegates, and edge

1. [ExecuTorch quantization docs](https://docs.pytorch.org/executorch/stable/quantization-overview.html) — prepare/convert/deploy path.
2. [ExecuTorch backend delegation](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html) — partition/delegate contract.
3. [PyTorch quantization docs](https://docs.pytorch.org/docs/stable/quantization.html) — baseline PT2E concepts.
4. [ExecuTorch backends source](https://github.com/pytorch/executorch/tree/main/backends) — inspect a delegate implementation.

## P59 — PyTorch source tree and build

1. [PyTorch from-source build](https://github.com/pytorch/pytorch#from-source) — build a pinned revision.
2. [PyTorch repository](https://github.com/pytorch/pytorch) — orient around `torch`, `aten`, `c10`, and `test`.
3. [PyTorch developer wiki](https://github.com/pytorch/pytorch/wiki) — use design notes when available.
4. [PyTorch codegen tree](https://github.com/pytorch/pytorch/tree/main/torchgen) — trace generated native API plumbing.

## P60 — PyTorch testing, CI, and triage

1. [PyTorch contributing guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) — required local checks.
2. [PyTorch test tree](https://github.com/pytorch/pytorch/tree/main/test) — learn test naming and device parametrization.
3. [PyTorch issue tracker](https://github.com/pytorch/pytorch/issues) — practice reproducing/triaging a bounded issue.
4. [GitHub Actions docs](https://docs.github.com/actions) — understand CI jobs and artifacts.

## P61 — PyTorch contribution engineering

1. [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) — submit-ready expectations.
2. [PyTorch code of conduct and project norms](https://github.com/pytorch/pytorch/blob/main/CODE_OF_CONDUCT.md) — understand community expectations alongside contribution rules.
3. [PyTorch developer discussion forum](https://dev-discuss.pytorch.org/) — research design context before proposing changes.
4. [Good first issues](https://github.com/pytorch/pytorch/contribute) — select a scoped, reproducible contribution.

## P62 — PyTorch ecosystem capstone

1. [PyTorch ecosystem](https://pytorch.org/ecosystem/) — map projects and ownership boundaries.
2. [PyTorch roadmap](https://github.com/pytorch/pytorch/wiki/Roadmap) — choose a current area deliberately.
3. [PyTorch contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) — turn the capstone into an upstream-quality change.
4. [PyTorch repository](https://github.com/pytorch/pytorch) — use issue, test, benchmark, and source evidence for the final artifact.
