# PyTorch Systems, Compilers, and Deployment: From Tensor Semantics to Upstream Contribution

> **Research snapshot:** 28 July 2026. PyTorch, TorchAO, torch.compile,
> ExecuTorch, and deployment APIs evolve quickly. Pin the version, commit, and
> hardware target used for every lab before treating an observation as general.

This is a separate, build-first companion to
[Inference Engineering: From Transformer Math to Upstream GPU Systems](../inference-engineering-curriculum/README.md).
It is designed for an engineer who can already build GenAI applications but
wants to become fluent enough in the PyTorch ecosystem to answer questions such
as:

- Why did this view, dtype, layout, stream, or aliasing choice change speed or
  correctness?
- What happens between Python `nn.Module` code and an ATen operator,
  autograd, a graph capture, an Inductor/Triton kernel, or a CUDA extension?
- How do I make a custom operation correct under eager execution,
  `torch.compile`, `torch.export`, FakeTensor, autograd, and dynamic shapes?
- When should an artifact travel through `torch.export`, ONNX/ONNX Runtime,
  Torch-TensorRT/TensorRT, AOTInductor, or ExecuTorch?
- How can I read the PyTorch source tree, reproduce a failure, write a test,
  and make an upstream-quality contribution?

The aim is not to memorize every PyTorch API. It is to make the framework
legible—from tensor storage and the dispatcher through compilers, native code,
distributed training, model optimization, and deployment—so you can build and
review systems with evidence.

## Start here

1. Read the [42-week flexible roadmap](ROADMAP.md) and the
   [module index](MODULE_INDEX.md).
2. Create the **separate** `torch-lab` repository described in
   [Lab environment](LAB_ENVIRONMENT.md). This directory contains curriculum
   documentation only; it deliberately does not create or modify that lab.
3. Read [How this connects to the inference curriculum](BRIDGE_TO_INFERENCE.md).
   It tells you which existing modules should be studied beside each PyTorch
   module, rather than making you relearn the same math twice.
4. Use the [concept map](CONCEPT_MAP.md) to see the semantic → compiler →
   runtime dependencies before skipping ahead to a framework branch.
5. Use [Framework guide](FRAMEWORK_GUIDE.md) to choose a deployment branch at
   the appropriate time. PyTorch is the spine; other frameworks are selected
   for a concrete artifact or runtime target.
6. Read the [PyTorch contribution path](CONTRIBUTING_TO_PYTORCH.md) during the
   first month. Contribution readiness is built continuously, not added after
   finishing a syllabus.

## The completion contract

A module counts only when you can do all six:

1. **Explain** the mechanism and its boundaries without slides or a copied
   notebook.
2. **Predict** behavior from shapes, strides, dtypes, aliases, device state,
   or graph constraints before running the program.
3. **Implement** a minimal correct version, with an eager reference oracle.
4. **Validate** with tests, numerical comparisons, and a failure case.
5. **Measure** the relevant metric: CPU/GPU time, memory, compile latency,
   graph count, communication, artifact size, or edge power/latency.
6. **Trace** the idea into an official implementation, test, issue, or source
   path and leave a source-walk note.

This matters especially for compiler and native-extension work: a fast-looking
benchmark with an invalid operator contract or a silently broken export is not
an optimization.

## What the course covers

There are **62 detailed concept modules**, six cumulative projects, and a
42-week standard route at your requested cadence of roughly 21 focused hours
per week. Treat the week count as a planning unit, not a deadline: source
builds, profiling, failed exports, and upstream review are productive work.

At three months, you should be able to build/debug a compact PyTorch
Transformer and navigate its framework behavior. At six months, you should be
able to own a compiler/custom-op or optimization investigation. The full
distributed/export/edge/contributor route is more realistically a 9–12 month
depth project; its milestone rubric is in
[Assessment and portfolio](ASSESSMENT_AND_PORTFOLIO.md).

| Phase | Modules | Outcome |
|---|---:|---|
| Systems habits and foundations | 01–06 | Reproducible environments, numerical and Python performance discipline, framework boundaries |
| PyTorch mental model | 07–21 | Fluent tensor/autograd/module/data/training code and a Transformer bridge |
| Model optimization and domains | 22–28 | AMP, tuning, TorchAO, vision, audio, multimodal, evaluation |
| Compiler and native extension stack | 29–42 | FX, Dynamo, AOTAutograd, Inductor, Triton, dispatcher, C++/CUDA, build and debug |
| Distributed PyTorch | 43–50 | Collectives, DDP/FSDP/DTensor, parallelism, checkpointing, profiling |
| Export, deployment, and upstream work | 51–62 | `torch.export`, ONNX, TensorRT, AOTInductor, ExecuTorch, PyTorch source and contribution |

## The deliberate framework hierarchy

The fastest way to become a framework tourist is to pursue every runtime in
parallel. Use this ordering instead:

1. **PyTorch core + torch.compile** — non-negotiable foundation. Learn tensor
   semantics, autograd, dispatch, graph capture, compilation, custom ops, and
   distributed APIs here.
2. **TorchAO + torchtune** — learn model optimization and fine-tuning where
   they naturally meet modern PyTorch. The legacy `torch.ao.quantization`
   surface is in migration, so this curriculum treats TorchAO as the forward
   path.
3. **Export and general serving** — `torch.export`, ONNX, ONNX Runtime, and
   AOTInductor. These teach the difference between a program contract and a
   runtime-specific artifact.
4. **NVIDIA deployment** — Torch-TensorRT, TensorRT, then TensorRT-LLM when
   the target is a stable NVIDIA serving path. Pair it with the existing
   inference course rather than trying to replace vLLM/SGLang.
5. **Edge** — ExecuTorch only after export concepts and quantization are solid.
   It becomes much clearer when you can reason about partitions, delegates,
   runtime ownership, and target hardware constraints.

`torchvision`, `torchaudio`, and multimodal model code appear as domain
exemplars; they are not detours. They make data transforms, stateful streaming,
operators, and export constraints concrete.

## Cumulative lab repository

Create a **separate Git repository** named `torch-lab`. A suggested layout:

```text
torch-lab/
├── README.md
├── pyproject.toml
├── environment/
│   ├── collect_env.sh
│   ├── versions.lock.md
│   └── build-notes.md
├── foundations/
├── pytorch_core/
│   ├── tensor_contracts/
│   ├── autograd/
│   ├── training_engine/
│   └── transformer/
├── optimization/
│   ├── amp/
│   ├── torchao/
│   └── torchtune/
├── compiler/
│   ├── fx/
│   ├── compile/
│   ├── triton/
│   └── custom_ops/
├── distributed/
├── export/
│   ├── torch_export/
│   ├── onnx/
│   ├── tensorrt/
│   └── executorch/
├── native/
│   ├── cpp_extension/
│   └── libtorch/
├── benchmarks/
│   ├── manifests/
│   ├── raw/
│   └── reports/
├── tests/
├── notes/
│   ├── source_walks/
│   ├── papers/
│   ├── export_dossiers/
│   └── failure_postmortems/
└── scripts/
```

Keep checkpoints, build caches, generated engine plans, and large trace files
out of Git. Every result should record the PyTorch commit/wheel version, CUDA
runtime and driver, GPU architecture, compiler versions, exact input-shape
distribution, warm-up policy, command, and raw artifacts.

## How to use a 21-hour week

| Session | Focus |
|---|---|
| Monday, 3 h | Derive the API/semantic contract; draw storage, graph, or execution diagrams |
| Tuesday, 3 h | Write the smallest eager implementation and tests |
| Wednesday, 3 h | Introduce a controlled failure; inspect errors, traces, and numerical deltas |
| Thursday, 3 h | Read production source and create a source-walk note |
| Friday, 3 h | Profile, compile, export, or distribute the same workload |
| Weekend, 6 h | Finish the deliverable, benchmark report, and one integration task |

There is an important rhythm behind this schedule: **eager oracle → semantic
test → graph/compiler view → native/runtime integration → measured claim**.
Do not start an extension, TensorRT engine, or multi-GPU run without a smaller
reference that can falsify it.

## Six cumulative projects

See [Projects](projects/README.md) for detailed milestones.

1. **Tensor and autograd microscope** — a testable suite that predicts views,
   aliases, gradients, and layout effects.
2. **PyTorch Transformer engine** — train and serve a compact decoder model
   with a KV-cache, profiling, and `torch.compile` variants.
3. **Optimization workbench** — AMP, TorchAO, fine-tuning, and quality/perf
   evidence for a real model component.
4. **Compiler-aware custom operator** — Python custom-op contract, FakeTensor,
   autograd, Triton/CUDA implementation, tests, and compile/export behavior.
5. **Portable deployment dossier** — export one model to ONNX Runtime and one
   NVIDIA or edge route, with parity and artifact analysis.
6. **Upstream-ready change** — a focused PyTorch or adjacent-ecosystem patch
   series with a reproducer, tests, documentation, and review-quality evidence.

## Local hardware context

Your RTX 5060 Ti (Blackwell / SM120) is an excellent systems lab, but it also
makes toolchain control non-optional. The FlashInfer NVFP4 compilation incident
is a first-class case study: a host-RAM failure during parallel `nvcc` work is
not a GPU-memory failure, and outer Ninja parallelism and inner compiler
threads must be bounded independently. Module 40 turns that real incident into
reusable build-debugging skill. See [Lab environment](LAB_ENVIRONMENT.md).

## Scope boundaries and the next horizon

This course makes you broad and deep enough to contribute to PyTorch-adjacent
work. It does not claim that 42 weeks completes CUDA compiler engineering,
formal numerical analysis, embedded systems, networking, model research, or
hardware design. The next strategic subjects after this core are listed in
[What not to miss](WHAT_NOT_TO_MISS.md): performance methodology, operating
systems and networking, compiler IR design, security/supply-chain discipline,
data quality/evaluation, and review culture are often the difference between a
clever optimization and a useful one.

## Source policy

Use official documentation, canonical repositories, source tests, standards,
and original papers first. Framework APIs change; read the versioned page and
record the commit you actually used. Every module has an ordered **Guided study
sources** section followed by three **Further study and source walks**—chosen
from practitioner explanations, official repositories, and implementation
material. The [source atlas](resources/SOURCE_ATLAS.md) completes that route
with production-source reading for all 62 modules. The recurring live resources
remain in [Primary sources](resources/PRIMARY_SOURCES.md).
