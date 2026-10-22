# Cumulative Projects

These are portfolio artifacts, not separate classes. Each project turns several
modules into one coherent, reviewable body of work. Keep the source in
`torch-lab`; retain this directory as the specification and evidence checklist.

| Project | When | Primary proof |
|---|---:|---|
| [01 — Tensor and autograd microscope](01-tensor-and-autograd-microscope.md) | Weeks 4–11 | You can predict aliasing, layout, gradients, and device behavior |
| [02 — PyTorch Transformer engine](02-pytorch-transformer-engine.md) | Weeks 10–15 | A tested Transformer ties PyTorch semantics to LLM architecture |
| [03 — Optimization workbench](03-optimization-workbench.md) | Weeks 12–25 | Quality, numerics, compile, and kernel claims have separate evidence |
| [04 — Distributed systems dossier](04-distributed-systems-dossier.md) | Weeks 26–31 | Topology, sharding, collectives, checkpointing, and measurement are explicit |
| [05 — Portable deployment dossier](05-portable-deployment-dossier.md) | Weeks 32–41 | Artifacts are validated across eager/exported/target runtime paths |
| [06 — Upstream-ready contribution](06-upstream-ready-contribution.md) | Weeks 1–42 | A narrow issue or improvement becomes a reproducible, tested patch |

## Project standard

Every project must include a README with a problem statement, architecture
diagram, environment manifest, run commands, test commands, results table, raw
artifact location, limitations, and next experiment. Preserve negative results
when they falsify a reasonable hypothesis; they demonstrate engineering
judgment.

Use the [benchmark report](../templates/BENCHMARK_REPORT.md),
[export report](../templates/EXPORT_REPORT.md), and
[source-walk template](../templates/PYTORCH_SOURCE_WALK.md) as recurring
evidence formats.
