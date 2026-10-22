# Open-Source Contribution Playbook

The fastest path into a high-performance runtime is not “find a hard kernel and
rewrite it.” It is to become reliable at reproducing, measuring, testing, and
communicating one bounded subsystem.

## Choose one primary project

Study all three runtime modules, then choose one primary codebase for the final
six to eight weeks:

| Project | Best fit |
|---|---|
| vLLM | Python/PyTorch engine design, scheduling, model integration, distributed serving, multiple kernel backends |
| SGLang | high-performance runtime scheduling, RadixAttention/prefix reuse, structured generation, kernels, distributed and multimodal systems |
| llama.cpp/ggml | C/C++, explicit compute graphs, GGUF/quantization, CPU and heterogeneous backends, edge/local inference |

Adjacent contribution targets include PyTorch, Triton, CUTLASS, FlashInfer,
Transformers, and model-specific repositories. Pick the repository where your
reproducer and hardware can provide meaningful evidence.

## Contribution ladder

### Level 0 — establish local truth

- Build from source at a pinned commit.
- Run the project's prescribed tests.
- Reproduce one published example or benchmark.
- Map one request from public API to model execution and output.
- Write down all fallback and dispatch decisions observed on your hardware.

Do not open an issue until the result is reproducible from a clean command.

### Level 1 — evidence contribution

Good first contributions include:

- a minimal reproducer for an unreported failure;
- a missing unit or regression test;
- a benchmark correction;
- documentation discovered only by reading source;
- a clearer diagnostic that names an unsupported shape or architecture;
- a small portability fix on hardware you actually possess.

These changes teach review, CI, ownership, and repository conventions without
large design risk.

### Level 2 — bounded feature or bug fix

Examples:

- model config or tokenizer edge-case support;
- dispatch/fallback correction;
- scheduler accounting invariant;
- quantization metadata or loader fix;
- kernel tail-shape correctness;
- profiler range or metric improvement.

Require a failing test before the fix and a benchmark when performance could
change.

### Level 3 — model or operator support

Add a model only after producing:

- architecture and checkpoint-tensor mapping;
- reference outputs and deterministic tiny tests;
- weight-loading validation;
- attention/KV/position semantics;
- quantization and multimodal considerations;
- memory and performance baseline;
- supported-hardware and fallback matrix.

Add an operator only with dtype/shape/layout contracts, reference parity,
dispatch policy, fallback, build integration, and benchmarks over the relevant
shape distribution.

### Level 4 — critical-path optimization or design change

Before coding:

- search issues, PRs, roadmaps, and design discussions;
- confirm a maintainer wants the direction;
- write a design/RFC if the project expects one;
- identify code owners and test infrastructure;
- declare the workloads and SLOs;
- establish baseline traces and correctness;
- split enabling changes from the optimization.

A 10% microbenchmark win is not compelling if the path is 1% of request time.

## A reviewer-ready issue

Use this structure in your own words:

1. **Observed behavior:** exact, neutral, and bounded.
2. **Expected behavior:** specification, test, or reason.
3. **Environment:** sanitized manifest and pinned revisions.
4. **Minimal reproduction:** smallest command/code and input.
5. **Evidence:** logs, trace excerpt, backtrace, profiler result.
6. **Bisect or scope:** last known good, affected dispatch region, or
   unsupported cases.
7. **Proposed next check:** optional, clearly separated from evidence.

Do not paste enormous logs inline. Attach the full artifact and quote the
minimal relevant portion.

## A reviewer-ready performance PR

Include:

- correctness test that fails or is missing before the patch;
- baseline and treatment commits;
- exact workload matrix and hardware/software manifest;
- independent-run statistics;
- profiler evidence for the claimed mechanism;
- quality results where numerics change;
- performance and memory regressions checked outside the winning shape;
- fallback behavior;
- generated-code or compile-time consequences;
- limitations and follow-up work.

Keep unrelated refactors out of the patch.

## Repository-specific entry points

### vLLM

Start with:

- [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/)
- [vLLM repository](https://github.com/vllm-project/vllm)
- [Basic model implementation guide](https://docs.vllm.ai/en/latest/contributing/model/basic/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)

Practical ladder:

1. documentation or diagnostic;
2. model test/loader edge case;
3. scheduler/cache invariant or platform-specific bug;
4. model implementation;
5. kernel dispatch or operator;
6. engine/RFC-level change.

Read the current rules about DCO, PR titles, tests, AI-assisted contributions,
and adding/changing kernels before opening work. The contribution guide changes
more quickly than this curriculum.

### SGLang

Start with:

- [SGLang contribution guide](https://docs.sglang.io/docs/developer_guide/contribution_guide)
- [SGLang repository](https://github.com/sgl-project/sglang)
- [SGLang paper](https://arxiv.org/abs/2312.07104)

The project explicitly recommends documentation as an onboarding path and
separates runtime and kernel-package release steps. Follow the current guide
when a change crosses `sglang` and its kernel distribution; do not assume one
PR can update and consume a newly released kernel.

Practical ladder:

1. docs/source-walk correction;
2. fast unit or accuracy test;
3. model/runtime bug;
4. scheduler, RadixAttention, structured-output, or multimodal path;
5. `sglang-kernel` operator and subsequent integration.

### llama.cpp/ggml

Start with:

- [llama.cpp contribution guide](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- [llama.cpp repository](https://github.com/ggml-org/llama.cpp)
- [How to add a model](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)

This project has explicit expectations for reproducible bug fixes, regression
tests, local CI, perplexity, `llama-bench`, backend-operator checks, portability,
and new quantization types. Read its current AI-usage policy before interacting:
the policy requires disclosure and places restrictions on AI-authored project
communication. You remain responsible for understanding and manually reviewing
every line.

Practical ladder:

1. reproducer or backend-op test;
2. bug fix in a known operator/model path;
3. model architecture/convertor support;
4. quantization or backend optimization with perplexity/KL/performance evidence;
5. new operator/backend or cross-platform design.

## How to study a repository

Do a vertical slice before reading folders horizontally:

```text
CLI/API entry
 → request/input object
 → tokenizer/processor
 → scheduler
 → model runner
 → model layer
 → operator dispatch
 → kernel/backend
 → cache/memory state
 → sampler/output
```

For each boundary, record:

- input/output types and tensor shapes;
- ownership and lifetime;
- sync/async or process boundary;
- device transfer;
- dispatch condition and fallback;
- test that exercises it;
- metrics/logging;
- likely maintainer from history/CODEOWNERS.

Then repeat horizontally for one subsystem, such as block allocation or
quantized linear dispatch.

## Finding useful work without chasing labels

“Good first issue” labels can be stale or contested. Better signals are:

- a maintainer-confirmed reproducer;
- recently modified code with missing edge tests;
- documentation that contradicts current behavior;
- a supported model/hardware combination you can actually test;
- profiler evidence of a localized regression;
- a TODO referenced by an active issue or roadmap;
- a small unreviewed gap in a recently merged feature.

Before claiming an issue, check active PRs and ask a narrowly framed technical
question. Never ask maintainers to design the entire contribution for you.

## Contribution dossier

For every candidate, create:

```text
contributions/<project>/<topic>/
├── upstream.md       issue, discussion, owners, project rules
├── reproducer/       minimal code and command
├── baseline/         environment and raw results
├── source-walk.md    call path and invariants
├── design.md         scope, alternatives, compatibility
├── tests.md          correctness and regression matrix
└── results.md        performance/quality evidence
```

Even if the patch is not merged, this dossier is a strong engineering artifact.

## Final readiness check

You are ready to open the PR when you can answer:

- What exact invariant or workload improves?
- Which supported behavior might regress?
- Why is this the correct layer for the change?
- How does dispatch select it and fall back?
- What test fails before and passes after?
- What profiler evidence supports the mechanism?
- Can another contributor reproduce the result?
- Have you followed the repository's current contribution and AI-use rules?
- Can you explain every line without an assistant?

