# 42-Week Flexible Roadmap

This is the standard route for approximately 21 focused hours each week. It is
purposefully not a race: a blocked compiler trace, failed native build, or
upstream review cycle should be written up and counted as learning, not skipped.
If you prefer a steadier pace, run each listed week across 1.25–1.5 calendar
weeks; that produces a 12–15 month route with more contribution time.

Before Week 1, spend one setup sprint on [Lab environment](LAB_ENVIRONMENT.md),
create `torch-lab`, and make a baseline `torch.utils.collect_env` capture. Do
not install every deployment runtime up front. Introduce each one when its
module requires it.

| Week | Modules | Main build / evidence | Existing inference bridge |
|---:|---|---|---|
| 1 | 01, 02 | Environment manifest; a controlled CUDA/Python failure and recovery note | 01, 02, 43 |
| 2 | 03, 04 | Python microbenchmark; dtype and reduction-error notebook | 03, 04, 05 |
| 3 | 05, 06 | Unit-test/benchmark harness and framework selection memo | 05, 44 |
| 4 | 07, 08 | Storage/stride visualizer and shape-property tests | 03, 36 |
| 5 | 09, 10 | Device/dtype/stream experiments and a tiny autograd engine | 04, 36, 41 |
| 6 | 11, 12 | State-dict/hook exercises; dispatcher source walk | 06, 41, 42 |
| 7 | 13, 14 | Data pipeline with worker/memory measurements; resumable optimizer state | 02, 44 |
| 8 | 15, 16 | Determinism matrix and checkpoint compatibility dossier | 02, 28, 43 |
| 9 | 17, 18 | `torch.func` transforms and a profiler/memory baseline | 03, 41, 44 |
| 10 | 19, 20 | Minimal training engine and compact decoder-only Transformer | 06–12 |
| 11 | 21 + Project 1 review | KV-cache-aware PyTorch decoding path; source walk | 10, 12, 29–34 |
| 12 | 22, 23 | AMP stability report; one torchtune recipe reproduced and modified | 04, 25, 26 |
| 13 | 24 | TorchAO quantization/sparsity experiment with quality and latency evidence | 18–28 |
| 14 | 25, 26 | Vision and streaming-audio mini-apps with profiling | 15, 44 |
| 15 | 27, 28 + Project 2 review | Multimodal path and evaluation protocol | 15, 17, 44 |
| 16 | 29, 30 | FX graph rewrite and `torch.compile` baseline | 39, 41, 42 |
| 17 | 31 | Guard/recompilation trace for a realistic shape distribution | 34, 42, 44 |
| 18 | 32 | AOTAutograd/functionalization source walk and mutation test suite | 41, 42 |
| 19 | 33, 34 | Inductor trace analysis and a Triton kernel parity benchmark | 37–42 |
| 20 | 35 | A compiler-safe Python custom op with `opcheck` | 41, 42 |
| 21 | 36 | C++/CUDA extension with correctness, build, and ABI notes | 36–43 |
| 22 | 37 | Dispatcher/Autograd registration source walk | 41, 42 |
| 23 | 38 | Operator lifecycle map from YAML/schema to tests | 41–43 |
| 24 | 39, 40 | Libtorch exercise; controlled Ninja/nvcc concurrency recovery dossier | 36, 41, 43 |
| 25 | 41, 42 + Project 3 review | CUDA Graphs integration and a minimized compiler regression | 35–44 |
| 26 | 43 | Two-process collective laboratory with topology notes | 45 |
| 27 | 44 | DDP scaling/profiling report | 45, 46, 47 |
| 28 | 45 | FSDP memory-accounting and checkpoint recovery experiment | 45 |
| 29 | 46 | DeviceMesh/DTensor placement prototype | 45 |
| 30 | 47, 48 | Parallelism map and distributed-checkpoint fault drill | 13, 45 |
| 31 | 49, 50 + Project 4 review | NCCL trace and training-to-serving parallelism dossier | 44–47 |
| 32 | 51 | `torch.export` program with declared dynamic-shape contract | 17, 42 |
| 33 | 52, 53 | ONNX artifact and ONNX Runtime parity/performance report | 28, 44 |
| 34 | 54 | Torch-TensorRT/TensorRT engine comparison, if target hardware supports it | 35, 37, 44 |
| 35 | 55 | TensorRT-LLM integration/source exploration or a documented deferral | 34, 45–47 |
| 36 | 56 | AOTInductor packaged artifact and ABI/runtime inventory | 39–43 |
| 37 | 57 | ExecuTorch export, partition, and runtime architecture note | 24, 28 |
| 38 | 58 | Edge quantization/delegate case study; use emulator/host if no device exists | 18–28 |
| 39 | 59, 60 | Local PyTorch source build plan; focused test/CI triage exercise | 41–43 |
| 40 | 61, 62 | Contribution dossier and ecosystem capstone design review | 46–48 |
| 41 | Project 5 | Portable deployment dossier: reference, exported, and target-runtime parity | 28, 42, 44 |
| 42 | Project 6 / buffer | Upstream-ready patch, review rehearsal, and curriculum retrospective | 46–48 |

## Gate reviews

Do not advance merely because the calendar moved. Use these gates:

| Gate | After week | You should be able to do |
|---|---:|---|
| PyTorch fluency | 11 | Explain tensor storage, autograd, module state, data loading, reproducibility, and build a verified compact Transformer |
| Optimization fluency | 15 | Separate model quality, numerical stability, kernel/runtime performance, and trainability claims |
| Compiler/native fluency | 25 | Diagnose graph breaks and recompiles; write a compile-aware custom op; navigate a C++/CUDA extension build |
| Distributed fluency | 31 | Explain data/shard/device/parallelism placement and capture a communication performance trace |
| Deployment fluency | 40 | Choose and validate an export/runtime route from an artifact contract rather than a benchmark headline |
| Contributor readiness | 42 | Submit a narrow, reproduced, tested, documented change or have one ready for maintainer feedback |

## Interleaving with the inference curriculum

The two curricula are complementary rather than sequential. In the first 15
weeks, prioritize PyTorch semantics while doing the matched inference modules
only when a model or numerical concept is needed. During Weeks 16–25, pair
compiler/native modules with Inference 35–44. During Weeks 26–42, use the
serving, distributed, and OSS modules of the inference course as the systems
counterpart. The detailed pairing is in [BRIDGE_TO_INFERENCE.md](BRIDGE_TO_INFERENCE.md).

## When to slow down

Add an extra week, never skip the evidence, when any of these happens:

- a numerical parity test fails without a root-cause explanation;
- a `torch.compile` result has unexpected graph breaks or uncontrolled
  recompilation;
- an exported artifact changes semantics, only supports a narrower dynamic
  contract, or depends on undeclared runtime libraries;
- a native extension works only in the current shell/cache and cannot be
  reproduced from a clean environment;
- a distributed result has no topology, synchronization, or per-rank evidence;
- an apparent speedup lacks warm-up policy, shape distribution, baseline, or
  correctness comparison.

That discipline is what turns framework proficiency into systems engineering.
