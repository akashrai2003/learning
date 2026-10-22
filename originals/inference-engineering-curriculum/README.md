# Inference Engineering: From Transformer Math to Upstream GPU Systems

> **Research snapshot:** 24 July 2026. Papers and stable principles remain
> useful; re-check live build, model-support, contribution, CI, and AI-use
> policies before acting upstream.

This is a build-first curriculum for becoming an independent LLM and
multimodal inference engineer. It is aimed at an engineer who already knows how
to use models and build agentic applications, but wants to understand the stack
deeply enough to:

- derive the math and reason about numerical error;
- implement reference model components without relying on `transformers`;
- quantize and validate models rather than only invoking a quantization CLI;
- write, profile, and integrate Triton and CUDA kernels;
- debug CUDA architecture, JIT, linker, ABI, and build-system failures;
- explain scheduler, KV-cache, attention, and distributed-serving trade-offs;
- navigate and contribute useful changes to vLLM, SGLang, llama.cpp/ggml, and
  adjacent projects.

The course has **48 concept modules**, six cumulative capstones, and two pacing
options. The normal route is 24 weeks at approximately 21 focused hours per
week. Six months is enough to become a strong, independently productive
practitioner; “complete mastery” of compilers, GPU architecture, numerical
methods, and distributed systems remains a longer career project.

## Start here

1. Read the [24-week roadmap](ROADMAP.md).
2. Study the [dependency map](CONCEPT_MAP.md); it explains why the order is not
   simply “all architecture, then all CUDA.”
3. Create the separate working repository described in
   [Lab environment](LAB_ENVIRONMENT.md).
4. Use the [local RTX 5060 Ti track](LOCAL_RTX_5060TI_TRACK.md) to map the
   modules onto the hardware already available.
5. Use the [module index](MODULE_INDEX.md) to begin Module 01.
6. Start a benchmark ledger using the
   [benchmarking protocol](BENCHMARKING.md) before optimizing anything.
7. Read the [open-source contribution playbook](OPEN_SOURCE.md) by Week 4, not
   at the end.
8. After the core, choose one of the
   [adjacent specialization tracks](EXTENSIONS_AFTER_CORE.md).

## The completion contract

A module is complete only when all six statements are true:

1. **Explain:** you can teach its central mechanism with tensor shapes and no
   slides.
2. **Derive:** you can reproduce the key equation, cost model, or invariant.
3. **Implement:** you have a clear reference implementation with unit tests.
4. **Measure:** you benchmark the relevant latency, throughput, memory, error,
   or quality metric.
5. **Inspect:** you locate the corresponding implementation in at least one
   production repository.
6. **Defend:** you pass the module's mastery gate without looking at the notes.

This prevents a common failure mode: accumulating papers and vocabulary without
being able to make, measure, or review a systems change.

## Cumulative lab repository

Create a separate Git repository named `inference-lab`. The curriculum files
are documentation; your lab is the portfolio:

```text
inference-lab/
├── README.md
├── pyproject.toml
├── environment/
│   ├── collect_env.sh
│   └── known-good.md
├── reference/
│   ├── tensor.py
│   ├── transformer.py
│   ├── attention.py
│   └── decoding.py
├── quantization/
├── kernels/
│   ├── triton/
│   ├── cuda/
│   └── extensions/
├── engine/
│   ├── cache/
│   ├── scheduler/
│   └── distributed/
├── benchmarks/
│   ├── manifests/
│   ├── results/
│   └── reports/
├── tests/
├── models/
│   └── architecture_notes/
└── notes/
    ├── papers/
    ├── source_walks/
    └── failure_postmortems/
```

Keep model weights and generated build caches outside Git. Every experiment
should record the Git commit, model revision, driver, toolkit, framework,
kernel-library versions, command, warm-up policy, and raw result.

## Course phases

| Phase | Modules | Main outcome |
|---|---:|---|
| Orientation and foundations | 01–05 | Reproduce environments, trace shapes, reason about precision and hardware limits |
| Transformer architecture | 06–17 | Implement a decoder and dissect modern dense, MoE, multimodal, long-context, and hybrid models |
| Quantization | 18–28 | Build and evaluate quantizers; understand formats, packing, calibration, and low-bit kernels |
| Attention and decoding systems | 29–34 | Implement online softmax, efficient attention, paged caches, sampling, speculation, and batching |
| GPU and compiler engineering | 35–44 | Write/profile kernels and understand CUDA, Triton, CUTLASS, custom ops, compilers, and JIT builds |
| Distributed runtimes and OSS | 45–48 | Reason about multi-GPU inference and contribute inside vLLM, SGLang, and llama.cpp |

The ordering intentionally alternates abstraction levels. For example, the
roofline model arrives before quantization, and quantized operator integration
arrives before CUDA programming. This lets each lower-level technique answer a
real bottleneck rather than becoming an isolated programming exercise.

## Standard week

At the requested cadence—three focused hours on weekdays plus six hours across
the weekend—use this loop:

| Session | Work |
|---|---|
| Monday, 3 h | Derive the mechanism; make shape tables and a one-page mental model |
| Tuesday, 3 h | Write the simplest correct reference implementation |
| Wednesday, 3 h | Add property tests and numerical comparisons |
| Thursday, 3 h | Study production code and annotate the call path |
| Friday, 3 h | Optimize or integrate one layer; profile before and after |
| Weekend, 6 h | Complete deliverable, benchmark matrix, paper note, and retrospective |

Two modules share the week. Usually one is theory/architecture and the other is
systems/implementation. When a module is especially large, use the weekend to
finish its mastery gate rather than racing ahead.

## What to build

The six [capstones](projects/README.md) are cumulative:

1. a framework-light decoder-only Transformer;
2. a quantization and quality workbench;
3. an optimized low-bit linear and attention kernel suite;
4. a small paged, continuously batched serving engine;
5. a modern-model port with an architecture and performance dossier;
6. a reviewed upstream contribution or an upstream-quality patch series.

By the end, the goal is not a toy “LLM from scratch” notebook. It is a tested
body of work showing numerical reasoning, profiling discipline, systems design,
GPU implementation, and code-reading ability.

## Rules that make the course effective

- **Correctness precedes speed.** Preserve a slow oracle for every optimized
  operator.
- **Profile hypotheses, not vibes.** State what resource should be limiting
  before collecting metrics.
- **One variable per comparison.** Do not change model, prompt mix, concurrency,
  precision, and engine flags together.
- **Distinguish prefill and decode.** They have different shapes, arithmetic
  intensity, bottlenecks, and useful optimizations.
- **Read source every week.** Papers describe ideas; production constraints live
  in tests, dispatch logic, build files, and fallback paths.
- **Keep failure postmortems.** A failed `sm_120` JIT build can teach memory
  pressure, Ninja scheduling, compiler flags, cache layout, and ABI boundaries
  better than a successful one-line install.
- **Contribute narrowly first.** Reproducers, tests, documentation, benchmark
  fixes, and small model ports establish trust before kernel rewrites.

## Source policy

Technical readings prioritize original papers, specifications, official
documentation, and canonical repositories. Blogs are used only when written by
the project or authors and when they add implementation context. Each module
contains its own source list; [Primary sources](resources/PRIMARY_SOURCES.md)
collects the recurring references.

Links and repository layouts evolve. At the start of a module, record the
repository commit or documentation version you actually studied.
