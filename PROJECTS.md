# Connected projects

Use one working repository, suggested name **systems-lab**, for these related artifacts. The original curricula propose several separate labs; that separation is optional here. Reuse tests, fixtures and measurement conventions. The four original curriculum directories remain your source library.

```text
systems-lab/
  README.md
  environment/          # versions, hardware, known-good commands
  reference/            # tensors, tiny training loop, decoder, cache
  native/               # C++, CMake, bindings and custom operators
  quantization/         # qparams, packing, format contracts
  kernels/              # Triton/CUDA and their reference comparisons
  serving/              # runner, real-runtime adapters, separate simulator
  deployment/           # export, ONNX, edge/mobile experiments
  fixtures/             # small public/synthetic inputs and expected behavior
  tests/
  benchmarks/           # manifests, raw results, report generation
  reports/              # readable findings and case studies
  notes/                # daily logs, paper notes, source walks, failures
  contributions/        # reproduction instructions and patch dossier
```

This is a proposed structure, not a prebuilt lab. Keep weights, large datasets, generated binaries and build caches outside source control. A contribution can use a separate upstream checkout while its evidence is indexed here.

## A — Serving benchmark and reliability lab

**Start week 1; useful first release week 4; extend through week 37.** This is the early career artifact. It wraps an established runtime; it does not require implementing a server first.

Start with one model, fixed prompts, structured raw records and explicit errors. Extend to controlled context/concurrency sweeps, cancellation, per-stage timings and operating-point reports. Reuse the same schema for deployment comparisons. A simulator added in week 23 is a distinct educational component; its results must be labeled simulated.

Done when a new reader can run one small supported experiment, regenerate a report and understand failed requests. Include model/runtime revisions, request counts, actual token counts where available, timing definitions and limitations. The project can be complete without beating another runtime.

Primary original specification: [career public project](originals/ai-career-acceleration/guides/public-project.md). The [mini-serving engine](originals/inference-engineering-curriculum/projects/04-mini-serving-engine.md) provides deeper extension work; a production-like custom engine is not required in this plan.

## B — PyTorch semantics and reference decoder

**Weeks 2–9; reused thereafter.** Begin with layouts, autograd, state/checkpoints and a tiny training loop. Build one dense decoder with normalization, FFN, causal attention, positions, GQA and cache. Select a compatible small checkpoint; do not attempt universal architecture support.

Done when causal invariants, full-versus-cached parity, position resets, serialization and tokenizer boundaries have meaningful checks. Intermediate comparison should identify the first incorrect operation. A small native linear/norm path connects the same oracle to C++.

Original projects: [tensor/autograd microscope](originals/pytorch-systems-and-deployment-curriculum/projects/01-tensor-and-autograd-microscope.md), [PyTorch Transformer engine](originals/pytorch-systems-and-deployment-curriculum/projects/02-pytorch-transformer-engine.md), [reference decoder](originals/inference-engineering-curriculum/projects/01-reference-decoder.md). Complete educational C++ model execution is an extension beyond the selected native operations.

## C — Quantization, compiler and kernel workbench

**Weeks 11–22, with buffer 12 and later repair week 24.** Add scalar INT8/INT4 references, bit-exact packing, calibration/evaluation separation and one supported TorchAO path. Compile the existing modules, diagnose guards, then implement a small Triton/CUDA operator and integrate it through a declared custom-op contract.

Done when each fast path has a preserved oracle, supported shape/dtype rules, boundary tests and raw benchmark records. Compare error, quality, actual bytes and end-to-end latency separately. One well-understood kernel is sufficient; a library of competitive kernels is not the required scope.

Original projects: [quantization workbench](originals/inference-engineering-curriculum/projects/02-quantization-workbench.md), [kernel suite](originals/inference-engineering-curriculum/projects/03-kernel-optimization-suite.md), [PyTorch optimization workbench](originals/pytorch-systems-and-deployment-curriculum/projects/03-optimization-workbench.md).

## D — Deployment and systems capstone dossier

**Weeks 25–34, release/assessment in weeks 37–39.** Gather production-runtime source evidence, local distributed semantics/recovery experiments and one verified export route. Add a small edge/mobile slice if hardware is available. Make one focused model/config/input adaptation in week 32.

Choose the final emphasis in week 33:

| Emphasis | Minimum end product | Optional expansion |
|---|---|---|
| Serving/ML systems — default | Reproducible real-runtime benchmark and reliability lab, one optimized/exported component and deployment decision | Fully custom scheduling server or multi-GPU scaling study |
| Edge specialization | Small native/ExecuTorch target with lifecycle and budget checks; host fallback clearly labeled if no device | Full offline document app, on-device embeddings or vendor NPU |

For either emphasis, include a tiny document/retrieval slice only if it clarifies your existing experience and fits the schedule. Use public/synthetic text; an exact retrieval baseline is enough. A benchmark tool can itself be the primary product. Do not build a second large application solely to satisfy a course checkbox.

Done when there is a reproducible success, a handled failure, a recovery demonstration, quality evidence and a documented memory/latency tradeoff. The distributed dossier may contain CPU two-process experiments and analytical models; they are not evidence of multi-GPU scaling.

Original projects: [distributed dossier](originals/pytorch-systems-and-deployment-curriculum/projects/04-distributed-systems-dossier.md), [portable deployment](originals/pytorch-systems-and-deployment-curriculum/projects/05-portable-deployment-dossier.md), [modern-model port](originals/inference-engineering-curriculum/projects/05-modern-model-port.md), [edge capstone](originals/edge-inference-roadmap/guides/capstone.md).

## E — One upstream-quality contribution

**Scout by week 4–5; keep scope small; focused sprint week 35.** Choose one repository already relevant to your experiments. Useful work can be a regression test, small fix, clarified behavior or benchmark correction. Keep at most one active contribution initially.

Done when the problem is reproduced, the change is narrow, relevant checks have recorded results, and a reviewer can understand the before/after behavior. A maintainer's response or merge is outside your control. A ready-to-review dossier is a valid learning outcome. Opening a PR is your later action, not something performed while writing this roadmap.

Original guides: [career contribution workflow](originals/ai-career-acceleration/guides/open-source-workflow.md), [PyTorch contribution project](originals/pytorch-systems-and-deployment-curriculum/projects/06-upstream-ready-contribution.md), [inference contribution project](originals/inference-engineering-curriculum/projects/06-upstream-contribution.md).

## Reuse evidence

One controlled experiment can supply a test, a report, a short public explanation and an interview story. Write the experiment first. The explanation should follow the observed result, including a negative result; it should not dictate a desired speedup.
