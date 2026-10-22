# Capstone Projects

The capstones turn 48 modules into one engineering narrative. They are
cumulative: later projects reuse the reference model, quantizers, kernels,
benchmark protocol, and source-reading habits developed earlier.

| Project | Start | Review | Core evidence |
|---|---:|---:|---|
| [01 — Reference decoder](01-reference-decoder.md) | Week 2 | Week 9 | semantic correctness and shape/memory reasoning |
| [02 — Quantization workbench](02-quantization-workbench.md) | Week 9 | Week 14 | numerical, quality, memory, and performance trade-offs |
| [03 — Kernel optimization suite](03-kernel-optimization-suite.md) | Week 15 | Week 22 | profiler-led CUDA/Triton/CUTLASS optimization |
| [04 — Mini serving engine](04-mini-serving-engine.md) | Week 15 | Week 23 | cache, batching, scheduling, and service metrics |
| [05 — Modern-model port](05-modern-model-port.md) | Week 7 | Week 24 | architecture forensics and runtime integration |
| [06 — Upstream contribution](06-upstream-contribution.md) | Week 4 | Week 24+ | maintainer-ready issue, tests, patch, and evidence |

Each capstone must have:

- a written contract and explicit non-goals;
- a high-confidence oracle;
- automated correctness tests;
- reproducible benchmark manifests and raw results;
- a design note explaining rejected alternatives;
- a final retrospective distinguishing knowledge gained from code produced.

Do not merge all projects into one giant package. Keep boundaries clear enough
that a reviewer can understand each artifact independently.

