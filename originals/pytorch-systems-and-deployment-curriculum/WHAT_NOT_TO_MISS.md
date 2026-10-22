# What Not to Miss: Strategic Additions to the Journey

Your stated goal is unusually good: move from agentic/GenAI application work to
the point where you can reason about and improve the machinery below it. The
largest risk is not lack of topics; it is trying to learn every adjacent stack
at equal depth. Keep **PyTorch systems + LLM inference** as the central pair,
then cultivate these supporting competencies.

## High-leverage subjects outside the obvious framework list

| Subject | Why it changes your effectiveness | Where to weave it in |
|---|---|---|
| Linux, processes, memory, filesystems, ELF/dynamic linking | Native builds and production incidents are often host/toolchain problems, not model problems | Modules 02, 36, 40, 56, 59 |
| Modern C++ and build systems | PyTorch/ATen/CUDA work requires ownership, lifetimes, templates, ABI, CMake, and test discipline | Modules 36–40, 59–61 |
| Performance methodology | The ability to reject an invalid speedup is more valuable than a one-off kernel trick | Modules 05, 18, 33–34, 41, 49 |
| Numerical analysis | Precision, reductions, attention stability, quantization, and reproducibility are all numerical contracts | Modules 04, 10, 22, 24; Inference 04, 18–28 |
| Compiler design and IRs | FX/Dynamo/Inductor are easier once you understand SSA, alias analysis, shape constraints, scheduling, and lowering | Modules 29–42; later study MLIR/TVM/IREE |
| Distributed systems and networks | Collectives, topology, fault tolerance, bandwidth/latency, and backpressure appear in training and serving | Modules 43–50; Inference 34, 45–47 |
| Evaluation and data governance | An optimization that changes accuracy, safety, latency tails, or reproducibility is a product change | Modules 23–28, 51–58 |
| Security and supply chain | Model artifacts, custom extensions, remote code, licenses, and benchmark data are operational risks | Modules 02, 16, 56, 59–61 |
| Review and open-source communication | Narrow reproducers, tests, changelogs, and respectful iteration are how useful patches land | Start in Week 1; Modules 60–62 |

## Four possible depth tracks after the shared core

Choose one as a primary identity and one as a secondary literacy track. You can
become conversant in all four; deepest contribution requires focus.

1. **PyTorch compiler/operator engineer.** Deepen Dynamo, Inductor, Triton,
   dispatcher, codegen, C++/CUDA, and compiler regression triage. Best fit for
   PyTorch core contributions.
2. **LLM runtime/inference engineer.** Deepen attention/KV cache, quantization,
   scheduling, GPU kernels, vLLM/SGLang/llama.cpp, and distributed serving. This
   naturally leverages your current local NVFP4 work.
3. **Deployment and edge engineer.** Deepen export contracts, ONNX Runtime,
   TensorRT, AOTInductor, ExecuTorch, device profiling, model compression, and
   artifact release engineering.
4. **Model systems / training-at-scale engineer.** Deepen torchtune, FSDP2,
   DTensor, activation checkpointing, data pipelines, checkpoint reliability,
   evaluation, and multimodal training.

For the next 3–6 months, I would make **Track 2 primary** and **Track 1
secondary**. It matches your demonstrated curiosity about NVFP4, FlashInfer,
Ninja, CUDA, and serving while building the exact PyTorch/compiler literacy that
turns kernel curiosity into upstream-capable work. Add Track 3 as a bounded
artifact project, not a third concurrent specialty.

## Habits that compound faster than more reading

- Keep an experiment ledger where every result has a baseline, hypothesis,
  shape distribution, environment, raw data, and conclusion.
- Maintain a “failed but explained” portfolio. Your compiler/OOM postmortems
  will be unusually valuable in interviews and code review.
- Read one production test file every week. Tests reveal contract boundaries
  more reliably than high-level diagrams.
- Reproduce one small paper claim per month, then identify the condition under
  which it no longer holds.
- Use issue trackers as a curriculum: triage a bug, minimize it, locate the
  code path, write a failing test locally, and only then consider a fix.
- Become bilingual in Python and C++ early; add CUDA/Triton after you can test
  and profile the integration layer.

## Anti-goals

- Do not equate API familiarity with systems proficiency.
- Do not publish latency numbers without numerical/task parity and a declared
  input distribution.
- Do not chase every new model architecture before you can write a clear
  architecture-forensics note for one.
- Do not start with a giant PyTorch source build or a large kernel rewrite;
  begin with source navigation, test selection, docs, small reproducers, and
  narrow fixes.
- Do not let local single-GPU results silently stand in for a multi-GPU or edge
  deployment claim.

The common thread is simple: make each learning step produce a durable artifact
that someone else could run, inspect, and review.
