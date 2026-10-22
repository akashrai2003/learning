# Framework and Runtime Guide

PyTorch is the core learning target. Other frameworks are included because they
teach an interface boundary or solve a deployment target—not because every
project needs every runtime.

## Choose from the artifact outward

| Need | First route | What you learn | Do not assume |
|---|---|---|---|
| Train/fine-tune/research a model with flexible Python control flow | PyTorch eager, then `torch.compile` | Tensor semantics, autograd, graphs, kernels, debugging | Compilation automatically improves every workload |
| Low-bit / sparsity work in modern PyTorch | TorchAO | Quantized tensor representations, workflows, lowering, quality/perf tradeoffs | Smaller stored weights automatically improve end-to-end latency |
| LLM fine-tuning recipes that stay close to PyTorch | torchtune | Configurable training recipes, data/model integration, reproducibility | A recipe is a substitute for understanding optimizer/model state |
| Read and prototype large-scale native-PyTorch LLM training | TorchTitan | Modern parallelism composition and training-system source reading | A reference project replaces learning DDP/FSDP/DTensor contracts first |
| Interchange with broad toolchains / CPU or heterogeneous runtimes | `torch.export` → ONNX → ONNX Runtime | Program constraints, ONNX IR, execution providers, artifact validation | ONNX export supports arbitrary Python or preserves performance unchanged |
| NVIDIA inference deployment with stable graph segments | Torch-TensorRT / TensorRT | Engine building, tactics, profiles, precision, runtime bindings | A TensorRT engine is portable across arbitrary GPUs, drivers, or shapes |
| NVIDIA LLM runtime evaluation | TensorRT-LLM | LLM-oriented kernels, parallelism, runtime integration | It replaces the need to understand vLLM/SGLang scheduler/KV-cache design |
| Native deployment without a full Python runtime | `torch.export` → AOTInductor | Ahead-of-time packaged code, ABI and runtime ownership | Packaging removes all version/driver compatibility concerns |
| Mobile, embedded, or edge targets | ExecuTorch | EXIR, delegates, partitioning, backend constraints, edge quantization | Edge deployment can be bolted on after unvalidated server deployment |
| Classical/portable optimized CPU inference | ONNX Runtime (and selectively OpenVINO, vendor EPs) | Hardware-specific execution-provider boundaries | A model’s CUDA result predicts CPU/edge behavior |

## Learning order and stopping rules

1. Finish Modules 01–21 before treating a runtime export as a meaningful
   performance project. You need a trustworthy eager reference first.
2. Finish Modules 22–24 before making low-bit deployment claims. Quantization
   is an accuracy/performance/representation problem, not a flag.
3. Finish Modules 29–35 before wrapping a custom kernel for production. An
   operator’s schema, fake implementation, aliasing/mutation, and transforms
   are part of its correctness contract.
4. Enter one deployment branch at a time in Modules 51–58. Preserve a parity
   harness between every artifact and the eager oracle.
5. Add a second runtime only if it answers a different product/hardware
   question. A new benchmark backend alone is not enough reason.

## Frameworks deliberately not made core

This course acknowledges but does not pursue equal-depth mastery of JAX/XLA,
TensorFlow, TVM, OpenVINO, MLIR/IREE, llama.cpp/ggml, and OpenAI Triton
alternatives. They are valuable, but equal treatment would dilute the goal of
becoming a capable PyTorch contributor. Use them as **comparison lenses** after
the PyTorch core:

- **JAX/XLA:** learn when you want a functional, ahead-of-time compilation
  contrast to PyTorch’s eager-first/Dynamo approach.
- **TVM/IREE/MLIR:** learn when compiler IR and target lowering become a primary
  career focus; they are excellent follow-ons after Modules 29–42.
- **OpenVINO / CPU vendor stacks:** learn when Intel/CPU deployment is a real
  target. Pair with ONNX Runtime as a baseline rather than collecting runtimes.
- **llama.cpp/ggml:** already has a dedicated bridge in the inference course;
  use it to compare a framework-free tensor/runtime design with PyTorch.
- **JAX, TensorFlow, and Keras:** know enough to read papers/repositories and
  translate tensor semantics, but do not split implementation effort early.

## A runtime comparison protocol

Whenever you compare two routes, the result must include:

1. Model/checkpoint and preprocessing revision.
2. Input and dynamic-shape distribution—not one cherry-picked shape.
3. Exact output-parity policy (absolute/relative tolerance, top-k agreement,
   task metric, and nondeterminism caveat).
4. Precision and quantization details, including calibration data if relevant.
5. Warm-up, synchronization, concurrency, timing source, and hardware details.
6. Build/cache/artifact version information and any fallback segments.
7. Peak memory, artifact size, compile/build time, and steady-state latency or
   throughput.

Use [Export report](templates/EXPORT_REPORT.md) to make this routine.
