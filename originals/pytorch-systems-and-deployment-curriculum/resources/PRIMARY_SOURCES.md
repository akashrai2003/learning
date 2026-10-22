# Primary Sources and Canonical Repositories

Use the module-local sources first. This is a recurring library, grouped by the
question it answers. Record the revision you read; documentation and APIs move.

## PyTorch foundations and source

- [PyTorch stable documentation](https://docs.pytorch.org/docs/stable/)
- [PyTorch source repository](https://github.com/pytorch/pytorch)
- [PyTorch contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library (paper)](https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)
- [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html)
- [torch.library](https://docs.pytorch.org/docs/stable/library.html)
- [PyTorch C++ API](https://docs.pytorch.org/cppdocs/)
- [LibTorch Stable ABI](https://docs.pytorch.org/docs/main/notes/libtorch_stable_abi.html)

## Compiler, graphs, and custom operations

- [torch.compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
- [torch.export programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/programming_model.html)
- [TorchInductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor)
- [TorchDynamo source](https://github.com/pytorch/pytorch/tree/main/torch/_dynamo)
- [Triton language repository](https://github.com/triton-lang/triton)
- [Triton language documentation](https://triton-lang.org/main/index.html)
- [PyTorch 2 compiler paper](https://docs.pytorch.org/assets/pytorch2-2.pdf)

## Training, model optimization, and domains

- [torch.func](https://docs.pytorch.org/docs/stable/func.html)
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html)
- [TorchAO source](https://github.com/pytorch/ao)
- [PyTorch quantization migration note](https://docs.pytorch.org/docs/stable/quantization.html)
- [torchtune documentation](https://docs.pytorch.org/torchtune/stable/)
- [torchtune source](https://github.com/pytorch/torchtune)
- [TorchTitan source](https://github.com/pytorch/torchtitan)
- [torchvision source](https://github.com/pytorch/vision)
- [torchaudio source](https://github.com/pytorch/audio)

## Distributed PyTorch

- [torch.distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)
- [FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html)
- [DTensor documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [Distributed checkpoint documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

## Export and general-purpose deployment

- [torch.export documentation](https://docs.pytorch.org/docs/stable/export.html)
- [PyTorch ONNX documentation](https://docs.pytorch.org/docs/stable/onnx.html)
- [torch.export-based ONNX exporter](https://docs.pytorch.org/docs/stable/onnx_export.html)
- [ONNX specification and project](https://onnx.ai/)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [ONNX Runtime performance guide](https://onnxruntime.ai/docs/performance/)

## NVIDIA deployment

- [Torch-TensorRT user guide](https://docs.pytorch.org/TensorRT/user_guide/index.html)
- [Torch-TensorRT source](https://github.com/pytorch/TensorRT)
- [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [NVIDIA TensorRT-LLM documentation](https://docs.nvidia.com/tensorrt-llm/index.html)
- [TensorRT-LLM source](https://github.com/NVIDIA/TensorRT-LLM)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS source and documentation](https://github.com/NVIDIA/cutlass)

## Edge

- [ExecuTorch getting started](https://docs.pytorch.org/executorch/stable/getting-started.html)
- [How ExecuTorch works](https://docs.pytorch.org/executorch/stable/intro-how-it-works.html)
- [ExecuTorch runtime overview](https://docs.pytorch.org/executorch/stable/runtime-overview.html)
- [ExecuTorch delegates and partitioners](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html)
- [ExecuTorch source](https://github.com/pytorch/executorch)

## Papers carried over from the inference curriculum

For Transformer, attention, quantization, GPU-kernel, and serving papers, use
the curated primary-source lists in the
[inference curriculum resources](../../inference-engineering-curriculum/resources/PRIMARY_SOURCES.md).
The bridge document specifies when those papers are prerequisites here.
