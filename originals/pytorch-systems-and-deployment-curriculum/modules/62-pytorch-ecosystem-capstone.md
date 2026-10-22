# Module 62 — PyTorch Ecosystem Capstone

**Phase:** 9 — PyTorch source and contribution engineering  
**Time:** 24 focused hours, then ongoing refinement  
**Prerequisites:** Complete one coherent thread through Modules 20–24, 29–42, 43–58, and 59–61; do not begin by combining every framework.

## Existing inference curriculum bridge

This capstone joins [46 — vLLM internals and contributions](../../inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md), [47 — SGLang internals and contributions](../../inference-engineering-curriculum/modules/47-sglang-internals-and-contributions.md), and [48 — llama.cpp/GGML internals and contributions](../../inference-engineering-curriculum/modules/48-llamacpp-ggml-internals-and-contributions.md) with the PyTorch and deployment path. It should produce one durable evidence trail—not a broad collection of unconnected demos.

## Why this matters and outcomes

The portfolio that signals inference/framework depth is a chain of reasoning: a model and workload contract; a correct reference; numerical tests; traces; compiler/export artifacts; deployment evidence; source maps; and one contribution-quality improvement. The capstone turns your local 5060 Ti, prior NVFP4 work, and PyTorch study into such a chain while preserving appropriate deferrals for unsupported edge/multi-GPU targets.

You will finish with a public-safe repository/project dossier, a selected upstream path, an honest capability matrix, and the ability to explain the same model/operator across eager PyTorch, compiler/native code, runtime/export, and serving integration.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P62 — PyTorch ecosystem capstone**.

1. [PyTorch ecosystem](https://pytorch.org/ecosystem/) — learn the API or system contract.
2. [PyTorch roadmap](https://github.com/pytorch/pytorch/wiki/Roadmap) — study the underlying design, paper, or specification.
3. [PyTorch contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch developer notes](https://docs.pytorch.org/docs/stable/community/contribution_guide.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch source repository](https://github.com/pytorch/pytorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch developer discussion](https://dev-discuss.pytorch.org/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and scope selection

Choose **one** of these capstone tracks, then add only evidence that advances it:

1. **Compact decoder systems thread:** implement a tested GQA/RoPE/SwiGLU decoder in PyTorch; cached decode; AMP/TorchAO experiment; `torch.compile`/export analysis; one custom-op or kernel boundary; ONNX/AOTInductor portability dossier; vLLM/SGLang source walk. This is the best bridge to your inference curriculum.
2. **Operator-to-upstream thread:** take a precise PyTorch operator/compiler/export bug or missing diagnostic from actionable issue to reproducer, source map, test, narrow patch plan, and—only with maintainer direction—PR. Pair it with a standalone extension/reference benchmark so the semantics are independently understandable.
3. **Edge multimodal thread:** select a small vision/audio/text encoder, build eager/evaluation/export/ExecuTorch XNNPACK or target-delegate paths, run backend-specific quantization, and report artifact/binary/memory/quality/latency boundaries. A physical device is required before device performance claims.
4. **Serving artifact thread:** take a supported small model through a declared deployment path (ONNX Runtime, Torch-TensorRT, AOTInductor, or TensorRT-LLM) and compare it to an eager/serving baseline with a fully specified workload. Do not conflate it with an all-runtime leaderboard.

The key artifact is a claim graph: every performance number points to workload/config/trace; every numerical claim points to an oracle/tolerance; every deployment claim points to artifact/dependency/target evidence; every upstream claim points to an actionable issue and test. “Deferred” is a valid state for multi-GPU NCCL, mobile NPU, or unsupported SM120 engine paths.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/capstone/charter.md` with problem, selected track, user/workload, model/checkpoint/license, hardware constraints, success metrics, exclusions, safety/privacy policy, source revisions, and a 6–10 experiment sequence.
2. **Optimized:** construct `evidence_index.md` linking reference tests, numeric ledger, profiler traces, compiler/export graph dumps, build manifests, deployment artifacts, and comparison tables. Every claim must have a reproducible command or explicit deferral.
3. **Integrated:** make `release_dossier.md` and an upstream dossier. Run a clean-environment reproduction, external-style code review, and final capability matrix. Submit upstream only when the issue/maintainer/process conditions in Module 61 are met.

## Three focused sessions

1. **Charter and baselines:** freeze the scope, prepare one correctness oracle and one workload harness, then obtain baseline quality/memory/latency evidence. Do not tune before this exists.
2. **One deep vertical:** follow the chosen model/op through at least three layers—for example eager → `torch.compile` → native op, or eager → export → edge delegate. Diagnose one failure/constraint from raw evidence.
3. **Review and handoff:** remove accidental environment dependence, rerun reproducibility checks, write limitations/deferred items, and simulate a maintainer or production-reviewer reading the project cold.

## Experiments and deliverables

Commit `charter.md`, `evidence_index.md`, `capability_matrix.md`, `release_dossier.md`, `reproduce.sh` or documented commands, and an `upstream_dossier.md`. Include all versions/commits, model/data licenses, hardware/driver/toolkit, environment hashes, inputs/outputs/quality policy, measurements with raw samples, profiler traces, artifacts/hashes, source maps, test coverage, comparison inclusion/exclusion criteria, known failures, security/secrets review, and next contribution target.

## Common failure modes

- Combining LLM serving, edge deployment, custom kernels, distributed training, and an upstream PR into one untestable project.
- Publishing impressive throughput without a workload, warm-up, sampling, memory, or quality contract.
- Calling a local host-runtime result an edge or multi-GPU result.
- Hiding unsupported paths instead of recording an evidence-backed deferral.
- Making an upstream PR the capstone's success criterion rather than producing a maintainable, reviewable contribution package.

## Mastery gate

Give a 20-minute technical walkthrough from model contract to artifact/runtime/source contribution, answering: what is measured; what is proven correct; what constraints are captured; what bytes/collectives/kernels dominate; why this runtime was chosen; and what remains unverified. A reviewer must be able to reproduce the core claim from the repository without your private environment.

## Primary sources, code paths, and OSS bridge

- [PyTorch source](https://github.com/pytorch/pytorch)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [TorchAO](https://github.com/pytorch/ao)
- [torchtune](https://github.com/pytorch/torchtune)
- [ExecuTorch](https://github.com/pytorch/executorch)
- [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)

Choose the repository whose abstraction boundary matches the evidence you have. A focused PyTorch test, TorchAO quantization issue, ExecuTorch backend coverage patch, vLLM build diagnostic, or SGLang runtime regression can all be a strong capstone. Contribution quality and continued engagement matter more than landing a large feature.

## Next

Repeat the evidence loop on a new architecture or target: learn the contract, build an oracle, measure, inspect the source, and contribute at the smallest leverage point.
