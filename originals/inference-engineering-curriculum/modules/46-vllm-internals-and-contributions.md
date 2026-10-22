# Module 46 — vLLM Internals and Contributions

**Phase:** 7 — Distributed runtimes and open source  
**Pacing:** Week 23, paired with Module 45; three 3-hour sessions plus a scoped contribution proposal  
**Prerequisites:** Modules 28–45; quantized dispatch, attention/KV management, scheduling, custom ops/compiler/builds, profiling, and distributed execution

## Why this matters

vLLM joins an API server, multi-process engine, scheduler, paged KV manager, model runner, model implementations, compilation/CUDA graphs, kernels, distributed workers, sampling, and streaming output. Reading folders alphabetically obscures the contracts. A vertical request trace reveals which process owns state, where a fallback occurs, and where a small patch can be tested without redesigning the engine.

## Outcomes

You can:

- trace an online request from protocol parsing to streamed text;
- locate scheduling, KV allocation, worker/model execution, sampling, and output ownership;
- trace a checkpoint's model and quantization configuration into selected kernels;
- build a minimal reproducer and choose the narrowest responsible subsystem;
- prepare a contribution that follows current vLLM testing, DCO, review, and AI-assistance rules.

## Mental model: control plane, scheduled state, execution plane

The current V1 process architecture is usefully viewed as:

```text
client -> API server/frontend -> engine client
 -> EngineCore process: request state + scheduler + KV manager
 -> executor -> GPU worker(s) -> model runner -> model/kernels
 -> sampler/output -> engine/output processor -> streamed response
```

Exact transports and optional coordinator processes depend on configuration, so verify them at the pinned commit. The scheduler emits a token-level plan, not merely a list of requests. The KV manager maps logical request progress to physical blocks. The model runner converts scheduler output into GPU tensors/block tables, executes eager/compiled/graph paths, samples, and returns bounded outputs. Frontend detokenization and protocol formatting must preserve request identity, stop conditions, logprobs, tools, and cancellation.

For the learner's Gemma 4 NVFP4 case, add a second vertical slice:

```text
HF config/checkpoint metadata -> model registry/implementation
 -> compressed-tensors quant config -> layer quant methods
 -> linear kernel registry capability checks
 -> FP8 attention linears + NVFP4 MLP kernel
 -> JIT/precompiled artifact -> CUDA launch
```

A mixed checkpoint legitimately selects multiple kernels. The log's FlashInfer CUTLASS NVFP4 backend is not a generic fallback, and Triton attention can coexist. Capture structured dispatch evidence before proposing a kernel change.

## Implementation ladder

### 1. Reference

Write `inference-lab/notes/source_walks/vllm-request.md` while tracing one tiny offline and one online request at a pinned commit. Identify every process boundary, queue/message, state owner, input/output type, and cancellation/error path.

### 2. Optimized

Create `inference-lab/engine/vllm_probe.py` or a non-invasive patch in a disposable fork that adds structured ranges/logging around one boundary: scheduler decision, block allocation, model execution, or kernel dispatch. Validate that observation overhead is absent when disabled.

### 3. Integrated

Select one real issue from evidence, reduce it to the smallest test, locate owners and existing issues/PRs, and draft a design in `inference-lab/notes/contributions/vllm-proposal.md`. Implement locally only after confirming scope. Include regression tests and benchmarks when behavior or performance changes.

## Three 3-hour sessions

1. **Request vertical:** trace CLI/API, engine client/core, scheduler/KV state, executor/worker, sampler, and streaming response.
2. **Model/kernel vertical:** trace Gemma 4 loading and mixed FP8/NVFP4 dispatch through compile/JIT/cache/launch on SM120.
3. **Contributor loop:** reproduce an issue from clean state, search ownership/history, write the failing test, run targeted checks, and prepare a reviewer-ready patch plan.

## Experiments and deliverables

Commit to the learner lab:

- `notes/source_walks/vllm-request.md`;
- `notes/source_walks/vllm-nvfp4-dispatch.md`;
- `engine/vllm_probe.py`;
- `tests/test_vllm_reproducer.py`;
- `benchmarks/results/vllm-patch-baseline.json`;
- `notes/contributions/vllm-proposal.md`.

The trace must pin commit/version, model/checkpoint, configuration, process IDs/ranks, scheduler/KV transitions, selected attention/linear kernels, compilation mode, JIT artifact, and output. The proposal separates observed behavior, expected invariant, reproducer, test, mechanism, scope, benchmark, and limitations.

## Practical contribution ladder

1. Reproduce a documented issue and map ownership.
2. Improve a test, diagnostic, or systematic documentation gap.
3. Fix a bounded loader/model/scheduler/cache/platform edge case with a failing test.
4. Add a model/operator path with complete contracts and fallbacks.
5. Optimize a critical path only after profiler evidence; seek an RFC for architectural changes.

The current guide requires DCO `Signed-off-by`, prescribed PR-title categories, tests/docs, and an RFC for major architectural changes (currently described as over 500 non-test/config LOC). Kernel ops require correct schemas, meta functions, and `opcheck`.

## AI-assisted contribution rules

Read the live guide before every submission. vLLM forbids “pure agent” PRs: the human must review every changed line, validate end to end, and remain responsible. Non-trivial AI help must be disclosed in the PR and attributed with the requested commit trailer; AI-assisted changes still need full tests and documentation. The guide also discourages isolated busywork. These rules can change; the curriculum is not a substitute for the current repository text.

## Common failure modes

- Reading only `LLMEngine` and assuming it owns every V1 concern.
- Ignoring process/rank boundaries when interpreting logs.
- Patching a model to work around a scheduler or platform capability bug.
- Forcing one quant backend across a mixed-format checkpoint.
- Treating successful compilation as completed CUDA graph warm-up.
- Opening a performance PR without proving critical-path selection.
- Running only a broad end-to-end test without a small failing invariant.
- Submitting AI-assisted code without human understanding, disclosure, or attribution.

## Mastery gate

Without global search for each step, narrate one request through concrete current code areas and state owners, then trace one linear layer from checkpoint metadata to SM120 launch. Produce a minimal failing test and scoped patch proposal that another contributor can evaluate. Explain why the change belongs in that subsystem, what regressions it guards, and which current contribution/AI rules apply.

## Canonical code areas and OSS bridge

Start at `vllm/entrypoints/cli/` and `vllm/entrypoints/openai/`; follow `vllm/v1/engine/`, `vllm/v1/core/sched/`, KV managers/block pool, `vllm/executor/`, `vllm/v1/worker/`, `vllm/model_executor/models/`, quantization and linear-kernel registries, `vllm/platforms/`, and `vllm/compilation/`. Test directories mirror many of these ownership boundaries. Paths evolve, so pair source with the current architecture guide rather than relying on line numbers.

## Primary and authoritative sources

- [vLLM architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)
- [vLLM source](https://github.com/vllm-project/vllm)
- [vLLM contributing guide and AI policy](https://docs.vllm.ai/en/latest/contributing/)
- [vLLM collaboration/RFC policy](https://docs.vllm.ai/en/latest/governance/collaboration/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)

## Next

Module 47 repeats the vertical method in SGLang, centering scheduler state, RadixAttention prefix reuse, structured generation, and its separately released kernel package.
