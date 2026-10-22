# Module 16 — Serialization, Checkpoints, and Model Artifacts

> **Phase:** II — Reliable PyTorch systems  
> **Time:** 9–11 focused hours across three sessions  
> **Prerequisites:** Modules 11, 14–15 and basic filesystem integrity concepts

## Existing inference curriculum bridge

This module extends [Module 02 — Reproducible Stack and Toolchain](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), [Module 17 — Modern-Model Architecture Forensics](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md), [Module 24 — Low-Bit Formats, NF4, Codebooks, and GGUF](../../inference-engineering-curriculum/modules/24-low-bit-formats-nf4-codebooks-and-gguf.md), and [Module 43 — Build, JIT, Ninja, CMake, and ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md). A model file is a system artifact with architecture, trust, layout, and compatibility boundaries—not merely tensors on disk.

## Why this matters and outcomes

A checkpoint that loads but resumes incorrectly is a failed training artifact. A checkpoint loaded from an untrusted source can be a security incident. An inference artifact lacking tokenizer, quantization, config, or software provenance may produce plausible but wrong outputs. Robust serialization is therefore correctness, operations, and supply-chain work.

By the end, you can choose state_dict over whole-object serialization; build a complete training checkpoint inventory; load across devices deliberately; inspect missing/unexpected state; use weights-only loading safely; and package model artifacts with hashes, manifests, validation, and rollback-friendly writes.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P16 — Serialization, checkpoints, and model artifacts**.

1. [Serialization note](https://docs.pytorch.org/docs/stable/notes/serialization.html) — learn the API or system contract.
2. [torch.load security guidance](https://docs.pytorch.org/docs/stable/generated/torch.load.html) — study the underlying design, paper, or specification.
3. [Distributed checkpoint docs](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch examples](https://github.com/pytorch/examples) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch recipes](https://docs.pytorch.org/tutorials/recipes/recipes_index.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

PyTorch serialization stores Python objects and tensor storages. The recommended model boundary is a state dictionary of parameters and persistent buffers, reconstructed into known code. Saving a whole Module couples the artifact to import paths, class definitions, and Python pickling behavior; it is rarely the right portable or long-lived contract. A state dictionary is still not self-describing architecture: save the model configuration and code/revision identity beside it.

Since PyTorch 2.6, torch.load defaults to weights_only=True when a pickle module is not supplied. This restricted unpickler supports tensors, primitive/container state, and approved globals rather than arbitrary dynamic imports. It narrows remote-code-execution exposure but is not a complete security boundary: it does not eliminate denial-of-service, malformed-artifact, or downstream tensor-shape risks. Do not set weights_only=False for an artifact merely to make an error disappear. Only load legacy whole objects from sources you trust, and review any safe-global allowlist rather than treating it as a convenience switch.

A normal model load is: construct the intended module from trusted config; load a known state dictionary with an explicit map_location; inspect the result of load_state_dict; move/cast the module under an explicit device policy; then run shape/numerical smoke tests. strict=True should be the default for a matched artifact. If warm-starting or intentional migration needs strict=False, assert the exact allowed missing/unexpected key set. Loading to CPU first is often the safest way to control VRAM pressure and inspect metadata.

A training checkpoint is a state-transition snapshot: model, optimizer, scheduler, AMP scaler when applicable, RNG states, data/sampler state, epoch/global/microstep, configuration, metrics, and dataset/model/tokenizer revisions. It must also state whether the resume promise is exact or approximate. Optimizer parameter identity/order and scheduler timing matter; recreating an optimizer after loading a model can be necessary in particular state-assignment modes, so test the documented load path.

Artifact hygiene complements API choice. Store immutable manifest fields: format/version, hashes and sizes, architecture config, tensor dtype/layout/quantization metadata, tokenizer and chat-template revisions, framework/runtime versions, source revision, license/provenance, and expected input/output contracts. Write to a temporary location, fsync/validate where the deployment environment warrants it, compute hashes, then atomically publish a manifest pointer. Never overwrite the only known-good artifact.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/artifacts/roundtrip.py. Save/load a toy model state_dict with weights_only=True, explicit map_location, strict key checking, and output parity on fixed CPU input.
2. Prove that an in-memory state_dict is shallow by mutating the live model after capturing it; then make an immutable saved snapshot and test it.

**Optimized**

3. Add a complete training checkpoint containing model, optimizer, scheduler, RNG, sampler, counters, configuration, and metrics. Resume a deterministic tiny run and compare trajectory/state.
4. Make an intentional key migration. Load with strict=False only after asserting the expected key difference and documenting the migration.

**Integrated**

5. Build an artifact directory contract with manifest JSON, SHA-256 hashes, model/config/tokenizer files, compatibility policy, and an atomic publication procedure. Validate before serving.
6. Create an untrusted-artifact test: demonstrate a weights-only failure, explain why allowlisting requires review, and never execute an untrusted legacy pickle.

## Three focused sessions

1. **State-dictionary boundaries (3h):** compare whole-object and state-dictionary artifacts, inspect keys/storage, and prove load parity across CPU/GPU policy without assuming implicit moves.
2. **Faithful checkpoints (3h):** save at step t, resume, compare parameter/optimizer/scheduler/RNG evolution, and classify any mismatch.
3. **Artifact operations (3h):** implement manifest/hash validation and fault injection for truncated, mismatched, and stale artifacts.

## Experiments and deliverables

Commit a checkpoint schema, round-trip tests, exact/approximate resume report, artifact manifest example, compatibility matrix, and security decision note. Include key inventory, tensor bytes, hash, source/revision, required runtime, input contract, and load command. Keep artifacts themselves ignored unless they are intentionally tiny fixtures.

## Common failure modes

- Saving a full Module because it is shorter than defining an artifact contract.
- Passing weights_only=False for untrusted data to bypass an error.
- Ignoring missing/unexpected keys after strict=False.
- Saving only model weights when claiming resumable training.
- Loading directly onto a crowded GPU and confusing load OOM with model size.
- Reusing a mutable in-memory state_dict as “best model.”
- Publishing a partially written or unhashed artifact.

## Mastery gate

Recover a deterministic toy run from a checkpoint with verified model, optimizer, scheduler, RNG, and sampler state. Reject a corrupt/mismatched artifact before model execution. Explain the threat model and exact compatibility policy of each load path.

## Primary sources and canonical code paths

- [PyTorch serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [PyTorch torch.load API](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch nn.Module load_state_dict](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- Inspect [serialization.py](https://github.com/pytorch/pytorch/blob/main/torch/serialization.py), [module.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py), and serialization tests.

## OSS bridge and next

Strong contributions include a safe loading regression test, a precise compatibility error, a migration test, or documentation that discourages unsafe artifact loading. Do not publish or share an artifact without provenance and a hash. Module 17 then treats module state as an explicit function, enabling vmap, Jacobians, per-sample gradients, and advanced transformations.
