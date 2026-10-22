# Contribution Path: Becoming Useful in the PyTorch Ecosystem

The reliable path into PyTorch is not “read all source, then write a major
kernel.” It is repeated practice at turning an observed behavior into a small,
reproducible, tested, maintainable change.

## Start contribution work in Month 1

1. Follow the current [PyTorch contributing guidance](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
   and re-check the repository's current issue/PR, CLA, testing, and AI-use
   policies before opening anything. Policies and CI requirements can change.
2. Subscribe to one area: `module: pt2`, `module: inductor`, `module: triton`,
   `module: distributed`, `module: onnx`, `module: quantization`, `module:
   cpu`, or a closely related satellite repository such as TorchAO, torchtune,
   ExecuTorch, Torch-TensorRT, vLLM, or SGLang.
3. Read three recent merged PRs and three open issues in that area. Note scope,
   tests, reviewer questions, regressions, and release notes.
4. Take one issue through the full local loop—even if you never submit the
   patch: reproduce, minimize, locate, write a failing test, propose options,
   and document the environment.

## Source-reading ladder

| Stage | Objective | Output |
|---|---|---|
| Call-site | Find public API, examples, and relevant test | Call graph with inputs/outputs |
| Python boundary | Locate `torch/`, FX/Dynamo/Inductor, or distributed wrapper | State/shape/dispatch annotations |
| Schema/dispatch | Find operator schema, dispatch registrations, autocast/autograd/meta behavior | Dispatch-key matrix |
| Native kernel | Read ATen/CUDA/Triton implementation and tests | Correctness/perf invariants |
| Build/CI | Identify generated files, build targets, platform gates, and test selectors | Minimal local command + CI risk list |

Use [the source-walk template](templates/PYTORCH_SOURCE_WALK.md). Source walks
should name exact commits; `main` is a moving target.

## High-value first contributions

- **Reproducible bug reports.** A 15-line script, expected/actual result,
  version/environment, and reduced input often save maintainers more time than a
  speculative patch.
- **Regression tests.** Especially shape/dtype/device/autograd/compile/export
  combinations where the desired contract is clear.
- **Documentation and examples.** A sharp example that captures dynamic-shape,
  custom-op, extension, or deployment behavior can be a real contribution.
- **Benchmark correctness.** Improve measurement setup, remove an invalid
  synchronization assumption, or add a representative workload—only after
  maintainer alignment.
- **Narrow implementation fixes.** Start with code already covered by an
  existing failing test. Avoid changing unrelated style or infrastructure in
  the same patch.

## Contributions appropriate to your current path

Your Blackwell/SM120 and local NVFP4 work create realistic entry points, but
scope them carefully:

- document or minimize an architecture/JIT/native-build issue with a clean
  environment matrix;
- add a focused test or build/documentation improvement where support already
  exists but behavior is unclear;
- contribute a benchmark/reproducer to FlashInfer, TorchAO, Torch-TensorRT,
  vLLM, or SGLang after confirming it is not a duplicate;
- trace an operator/compile/export gap end-to-end before proposing a kernel.

Do not lead with a request that maintainers accept a large unmeasured SM120
optimization. Lead with a clear contract, a narrow workload, tests, and evidence
of where the current path breaks or leaves performance on the table.

## The upstream-quality patch checklist

- Problem scope, current behavior, expected behavior, and non-goals are clear.
- Reproducer is minimal and includes fixed seed/device/dtype/shape details.
- Test fails before and passes after the change; relevant adjacent tests pass.
- Public API/serialization/backward-compatibility effects are explicitly
  considered.
- Performance claims include a baseline, target workload, hardware/toolchain,
  correctness checks, variance, and cold/warm distinction.
- New operator paths account for schema, mutation/aliasing, FakeTensor/meta,
  autograd, vmap if relevant, `torch.compile`, and export behavior.
- Build changes name their compiler/toolchain/platform effect and do not rely on
  untracked shell state.
- Commit series is small, logically ordered, and readable independently.
- PR text is concise, explains testing, and invites review on the actual
  trade-off or uncertainty.

## Feedback is part of the curriculum

When a maintainer asks for a reduction, a missing test, a benchmark rerun, or a
different abstraction, record it in the contribution dossier. It teaches the
project’s real contracts better than a static guide. Do not treat a closed issue
or declined design as failure if the investigation was rigorous; convert it into
a source-walk and reuse the learning on the next narrow task.

## Project selection rule

Contribute where your evidence and motivation align:

| Interest | Likely homes |
|---|---|
| Compiler semantics/custom ops | PyTorch Dynamo/Inductor/dispatcher, Triton |
| Low-bit/quantization | TorchAO, PyTorch quantization/export, vLLM/SGLang/FlashInfer integration |
| LLM serving | vLLM, SGLang, FlashInfer, TensorRT-LLM, llama.cpp/ggml |
| Edge/export | PyTorch export/ONNX, ExecuTorch, Torch-TensorRT, ONNX Runtime |
| Training scale | torch.distributed, FSDP/DTensor, torchtune/torchtitan |

Read each project’s own contribution and security policies before acting. A
course cannot replace maintainer direction.
