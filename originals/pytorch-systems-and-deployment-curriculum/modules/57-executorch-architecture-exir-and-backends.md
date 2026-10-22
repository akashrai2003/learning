# Module 57 — ExecuTorch Architecture, EXIR, and Backends

**Phase:** 8 — Export and deployment  
**Time:** 18 focused hours  
**Prerequisites:** Modules 51–52, basic C++ build literacy, and willingness to work with an evolving edge stack.

## Existing inference curriculum bridge

Study this after [15 — Multimodal inference](../../inference-engineering-curriculum/modules/15-multimodal-inference.md), [17 — Modern model architecture forensics](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md), [28 — Quantized kernel and system integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), and [42 — Compiler stack: Dynamo, Inductor, and Triton](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md). It applies export/compiler concepts to constrained devices where binary size, memory arenas, and delegates are first-order requirements.

## Why this matters and outcomes

ExecuTorch is an inference-focused edge runtime and compiler flow, not a smaller Python server. It converts an exported PyTorch program through edge/backend dialects into a `.pte` program that a portable C++ runtime executes. It makes the deployment boundary vivid: an operator set, memory plan, runtime/kernel registration, delegate blobs, platform abstraction, and target device must all agree.

You will be able to explain the complete eager-to-edge flow, export and execute a small `.pte` model in the host Python runtime, inspect delegation/partitioning, choose an appropriate CPU/mobile backend, and prepare an on-device branch without pretending a desktop GPU test proves mobile behavior.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P57 — ExecuTorch architecture, EXIR, and backends**.

1. [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/) — learn the API or system contract.
2. [ExecuTorch runtime overview](https://docs.pytorch.org/executorch/stable/runtime-overview.html) — study the underlying design, paper, or specification.
3. [ExecuTorch repository](https://github.com/pytorch/executorch) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

ExecuTorch starts from a `torch.export` program in ATen dialect, transforms it to Edge dialect, optionally partitions/lower subgraphs to backend dialects, then emits an ExecuTorch program binary. The runtime loads `.pte` programs, maps operator names to registered C++ kernels, executes instructions, dispatches `call_delegate` regions to backend code, and manages planned memory. Export IR documentation describes the ATen → Edge → Backend progression; the API/graph representation is evolving, so record ExecuTorch and PyTorch revisions alongside each artifact.

A backend has an ahead-of-time preprocessing half and a runtime half. Its partitioner tags eligible nodes; lowering sends connected tagged subgraphs to preprocessing, yielding delegate-specific bytes/configuration. At runtime, those bytes are opaque to the core executor and invoked as delegate calls. A partitioner is therefore a hardware capability policy, not simply an optimizer pass. Audit which nodes remain portable kernels, which move to a delegate, and what happens at each boundary.

ExecuTorch's runtime design emphasizes bounded overhead and controllable memory: planned mutable tensor arenas, constants pointing into program data, selective kernel builds, and platform abstraction. That shifts the question from “does it run?” to “what exact runtime binary, delegate registrations, memory plan, and target ABI make it run?” The host Python runtime is excellent for parity and inspection, but target Android/iOS/Linux/embedded execution needs target-specific toolchain, backend, device, and measurement evidence.

## Implementation ladder in `torch-lab/`

1. **Reference:** make `torch-lab/executorch/model_contract.py` for a small MobileNet-like, audio, or compact decoder submodel. Reuse export boundary and numeric tests from Modules 51–52.
2. **Optimized:** create `export_pte.py` that calls the current `torch.export` → `to_edge_transform_and_lower` → `to_executorch` flow with a documented XNNPACK or portable target. Load/run the `.pte` through the host Python runtime and inspect methods/output.
3. **Integrated:** write `edge_architecture_dossier.md`. Draw the ATen/Edge/Backend/runtime route, list every node/delegate region, `.pte`/runtime build hashes, memory/binary budget, and a target-device validation plan. Target device unavailable is a deferred evidence branch, not a performance conclusion.

## Three focused sessions

1. **IR progression:** export a small model, dump every stage, and identify changes in operator namespace, layout/dim-order handling, and graph boundaries. Tie each change to an actual target requirement.
2. **Delegate audit:** lower with one backend partitioner; compare graphs before/after, enumerate covered/uncovered nodes, and test host-runtime parity for representative/boundary inputs.
3. **Runtime/build planning:** study the C++ runtime architecture and selective build. Design a target application’s runtime/kernel/delegate/memory layout before acquiring a phone or board.

## Experiments and deliverables

Commit `model_contract.py`, `export_pte.py`, `test_pte_parity.py`, graph dumps, `pte_manifest.json`, and `edge_architecture_dossier.md`. Record PyTorch/ExecuTorch/backend revisions, model/input contract, dialects/ops, partitioner/delegate versions, covered/uncovered nodes, `.pte` hash/size, host runtime output/tolerance, runtime build flags, intended target ABI, memory/binary budgets, and deferred target evidence.

## Common failure modes

- Treating `.pte` as a generic ONNX-like file without a runtime/kernel/delegate contract.
- Assuming host Python runtime success proves Android/iOS/embedded correctness or speed.
- Ignoring undelegated nodes and calling the whole model accelerated.
- Introducing custom ops without a target runtime registration/build plan.
- Using dynamic shapes without valid ranges compatible with memory planning.

## Mastery gate

Produce a `.pte` artifact that has eager/export/host-runtime parity, an inspectable dialect/partition map, and a target-runtime bill of materials. Explain exactly what will execute in core kernels versus a delegate and what target-device test remains before declaring deployment success.

## Primary sources, code paths, and OSS bridge

- [ExecuTorch getting started](https://docs.pytorch.org/executorch/stable/getting-started.html)
- [How ExecuTorch works](https://docs.pytorch.org/executorch/stable/intro-how-it-works)
- [Export IR specification](https://docs.pytorch.org/executorch/stable/ir-exir.html)
- [Runtime overview](https://docs.pytorch.org/executorch/stable/runtime-overview.html)
- [Backends, delegates, and partitioners](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html)

Read [ExecuTorch source](https://github.com/pytorch/executorch), especially `executorch/exir/`, `runtime/executor/`, and one backend's partitioner/preprocess/runtime implementation. Target a small missing-op, diagnostic, test, or documentation contribution after checking maintainer guidance; changing IR/delegate contracts requires much wider compatibility evidence.

## Next

Module 58 makes quantization and delegate coverage a measured edge deployment decision.
