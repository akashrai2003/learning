# Project 04 — Distributed Systems Dossier

**Window:** Weeks 26–31 · **Prerequisites:** Modules 43–50

Create a small multi-process laboratory that makes placement, collectives, and
failure behavior concrete. One GPU is enough to learn the contracts using CPU
process groups or local multi-process simulation; use multiple GPUs or a remote
cluster only when legitimately available.

## Required scope

- A two-rank collective exercise with explicit rank/world-size/group setup and
  a deadlock/error case explained.
- A DDP run that measures scaling, synchronization points, unused/conditional
  parameter behavior, and data-sharding correctness.
- An FSDP or sharding experiment with a memory-accounting table: parameters,
  gradients, optimizer state, activations, communication buffers, and
  checkpoint form.
- A DeviceMesh/DTensor placement sketch, including an explanation of `Shard`,
  `Replicate`, and `Partial` for your chosen operation.
- A distributed checkpoint/recovery drill and a topology/NCCL trace where
  hardware permits.
- A design translation from training partitioning to LLM serving tensor/pipeline
  /expert parallelism. Mark what does **not** transfer because traffic and cache
  behavior differ.

## Deliverables and gate

Produce a rank-by-rank timeline, placement diagram, memory ledger, benchmark
manifest, and a recovery postmortem. You pass when you can diagnose a hang or
wrong result through group membership, rank order, mesh consistency, device
selection, data partitioning, or collective mismatch rather than changing flags
at random.

**Inference bridge:** Inference 45–47, especially the distinction between
training all-reduce/shard traffic and serving prefill/decode/cache traffic.
