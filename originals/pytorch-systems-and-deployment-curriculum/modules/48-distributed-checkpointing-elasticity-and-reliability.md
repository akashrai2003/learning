# Module 48 — Distributed Checkpointing, Elasticity, and Reliability

**Phase:** 7 — Distributed systems foundations  
**Time:** 15 focused hours  
**Prerequisites:** Modules 16, 43–47, a reproducible training loop, and basic filesystem failure awareness.

## Existing inference curriculum bridge

This reinforces [02 — Reproducible stack and toolchain](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md), and [45 — Distributed inference parallelism](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md). Training recovery and serving availability are different systems, but both require an explicit state boundary, ownership map, and failure drill.

## Why this matters and outcomes

A model checkpoint is not necessarily a restartable training state. A sharded job must preserve model parameters, optimizer slots, scheduler/RNG/data cursor, metadata, and the meaning of its rank topology. An elastic launcher may restart every worker with a different rank/world size. If resume semantics are not tested under interruption, a successful save call proves almost nothing.

You will be able to decide which state belongs in a checkpoint, use PyTorch distributed checkpoint APIs cautiously, design rank-safe save/load tests, distinguish fault tolerance from elasticity, and write a recovery runbook that a collaborator could execute.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P48 — Distributed checkpointing, elasticity, and reliability**.

1. [Distributed checkpointing](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) — learn the API or system contract.
2. [TorchElastic](https://docs.pytorch.org/docs/stable/elastic/run.html) — study the underlying design, paper, or specification.
3. [FSDP state-dict docs](https://docs.pytorch.org/docs/stable/fsdp.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Define a checkpoint contract before choosing a library: training step; model/optimizer/scheduler/scaler state; RNG for Python/NumPy/Torch/device; data sampler/dataset cursor; parallelism topology; software/config/model hashes; and artifact completeness marker. A full state dict may be convenient for inference export but can create a rank-zero CPU/GPU spike. A sharded state avoids that gather but needs a storage layout and loader that understand sharding.

`torch.distributed.checkpoint` (DCP) coordinates state across ranks and its state-dict helpers can work with supported DDP/FSDP/DTensor parallelisms. The documentation currently labels some APIs experimental; version-pin imports and signatures, retain a migration test, and never assume an opaque shard directory is portable forever. Storage atomicity is also external: create a temporary checkpoint namespace, have all writers finish, validate a manifest/hashes, then publish a complete marker or atomic directory move supported by the actual filesystem.

`torchrun` fault-tolerant mode restarts a fixed worker group up to a limit. Elastic membership change stops existing workers and starts a new group with new `RANK`/`WORLD_SIZE`; code must not treat ranks as permanent identities. A restart is only correct if the program reloads the intended committed checkpoint and handles data progress deliberately. On local hardware, use two CPU Gloo workers and a controlled exception/kill to validate mechanics; do not claim distributed storage durability from a single local filesystem.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/reliability/checkpoint_contract.md` and `state_snapshot.py` for a single-process compact Transformer. Serialize all named state explicitly and test bitwise/reproducible continuation where the algorithm permits it.
2. **Optimized:** add `dcp_roundtrip.py` for two CPU ranks. Save a documented state dictionary, reload it into a fresh process group, and compare model/optimizer/data-step/RNG outcomes after one subsequent update.
3. **Integrated:** write `recovery_drill.sh` and `recovery_report.md`. Inject a worker failure at a specified step under `torchrun --max-restarts`, prove which checkpoint resumed, and list all state that remains intentionally non-durable.

## Three focused sessions

1. **State audit:** take a training interruption before/after an optimizer update; enumerate every difference needed to get the same next update. Add version/config/hardware metadata without making it a substitute for semantic data.
2. **Atomicity drill:** simulate an incomplete shard write, make the loader reject it via a manifest, then test a complete two-rank round trip in a fresh process.
3. **Restart drill:** deliberately fail one worker, capture launcher/rank logs, resume from the last committed state, and verify no silent duplicate or skipped data beyond your declared policy.

## Experiments and deliverables

Commit `checkpoint_contract.md`, `dcp_roundtrip.py`, `test_resume_equivalence.py`, `recovery_drill.sh`, `manifest_schema.json`, and `recovery_report.md`. Record checkpoint format/API revision, shard map, total bytes/write time, storage characteristics, commit protocol, hashes, configuration/model/data versions, restart trigger, rank/world-size before/after, sampler cursor policy, and numerical continuation tolerance.

## Common failure modes

- Saving model weights but omitting optimizer, scheduler, scaler, RNG, or data progress.
- Having all ranks write a full state to the same path.
- Calling a directory with partial shards a valid checkpoint.
- Assuming a restarted rank has the same identity or local device.
- Testing only load success instead of the next training update.

## Mastery gate

Run a two-rank interruption/restart exercise and demonstrate a correct subsequent update against a declared oracle. Explain why the chosen artifact is suitable for training recovery, inference loading, both, or neither. Identify the external storage guarantee your protocol actually relies on.

## Primary sources, code paths, and OSS bridge

- [PyTorch distributed checkpoint documentation](https://docs.pytorch.org/docs/main/distributed.checkpoint.html)
- [`torchrun` elastic launch](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [TorchElastic error propagation](https://docs.pytorch.org/docs/stable/elastic/errors.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)

Read `torch/distributed/checkpoint/`, `torch/distributed/elastic/`, their tests, and a pinned TorchTitan checkpoint configuration. Candidate contributions are exact failure/recovery regressions or documentation corrections supported by a reproducer; distributed persistence changes need maintainer agreement and multi-environment evidence.

## Next

Module 49 connects the collectives in these recovery-aware designs to physical topology and profiler evidence.
