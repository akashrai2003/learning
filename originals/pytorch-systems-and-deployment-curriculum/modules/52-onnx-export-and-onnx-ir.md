# Module 52 — ONNX Export and ONNX IR

**Phase:** 8 — Export and deployment  
**Time:** 15 focused hours  
**Prerequisites:** Module 51, Modules 06 and 29, and basic graph/serialization literacy.

## Existing inference curriculum bridge

Pair this with [17 — Modern model architecture forensics](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md), [28 — Quantized kernel and system integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), and [42 — Compiler stack: Dynamo, Inductor, and Triton](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md). ONNX makes the model/operator contract portable, but it does not erase kernel, packing, cache, or scheduler decisions from the inference course.

## Why this matters and outcomes

ONNX is an interchange IR with a versioned operator set, graph inputs/initializers, type/shape information, and optional custom domains. It is not “PyTorch serialized differently,” and a file passing `onnx.checker` is not automatically a correct or fast production deployment. Understanding the IR lets you tell exporter bugs, unsupported operations, dynamic-shape ambiguity, custom-op boundaries, and runtime limitations apart.

You will be able to use PyTorch's modern ONNX exporter, inspect and validate an ONNX model, reason about opset and external-data choices, maintain source-to-node provenance, and decide whether an unsupported operation needs a decomposition, custom translation, custom runtime op, or a different deployment path.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P52 — ONNX export and ONNX IR**.

1. [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html) — learn the API or system contract.
2. [ONNX IR specification](https://onnx.ai/onnx/intro/) — study the underlying design, paper, or specification.
3. [ONNX exporter tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

The modern PyTorch path is `torch.onnx.export(..., dynamo=True)`, which is current documentation's default/recommended route and is based on `torch.export.ExportedProgram`. Its exporter attempts to obtain an exported program and then translates normalized PyTorch operations to a chosen ONNX opset. Pin the exporter/ONNX/opset versions: semantics, support, diagnostics, and defaults evolve. The old `dynamic_axes` idiom and the newer `dynamic_shapes` contract are not interchangeable by assumption; use the current API for the exporter you actually invoke.

An ONNX `ModelProto` contains an IR version, opset imports, graph, metadata, and optional functions. A graph has values, nodes, initializers, and inputs/outputs; weights may be embedded initializers or external data. Symbolic dimensions communicate flexibility but do not mean every runtime or accelerator will compile every value in that range. Preserve exact names and source metadata so a bad ONNX node can be traced back to its PyTorch operation/module.

Custom operations have two contracts: exporter representation and runtime implementation. A custom domain node may make export succeed but will fail downstream unless the selected runtime has a compatible kernel/registration. Prefer supported ATen/operator decompositions when they preserve behavior; otherwise write a narrow custom translation and runtime plan. Never substitute an approximate node merely to make a graph checker pass.

## Implementation ladder in `torch-lab/`

1. **Reference:** extend the Module 51 eager/export contract with `onnx_reference_inputs.npz` and a manifest that fixes model, inputs, expected output names, opset, and shape policy.
2. **Optimized:** create `export_onnx.py` using the Dynamo/export path, serialize the returned program deliberately, run ONNX checker/shape inference where valid, and save the exporter report when a failure occurs.
3. **Integrated:** add `onnx_inspect.py` and `onnx_dossier.md`. Map every interesting ONNX node to a source PyTorch op, inspect inputs/initializers/external data, test fixed and dynamic boundary shapes, and document any custom domain/runtime requirement.

## Three focused sessions

1. **IR walk:** export a small MLP or compact attention block; print IR/opset imports, graph inputs versus initializers, node attributes, value shapes, and source metadata.
2. **Shape/opset experiment:** compare fixed and bounded dynamic exports at a pinned opset. Run checker and intended-runtime parity tests; interpret a failure as a contract discrepancy, not automatically an exporter defect.
3. **Custom-boundary decision:** choose one unsupported or custom PyTorch operation and write a decision record: decomposition, translation table, custom ONNX domain plus kernel, or no ONNX route. Include accuracy and maintenance costs.

## Experiments and deliverables

Commit `export_onnx.py`, `onnx_inspect.py`, `test_onnx_contract.py`, `model.onnx` or reproducible build instructions, `onnx_dossier.md`, and `export_report.md`. Record PyTorch/ONNX/ONNX Runtime versions, opset, exporter mode, model hash, external-data files/hashes, graph input/initializer lists, symbolic dimensions, checker/shape-inference results, node-source map, custom domains, and per-shape parity.

## Common failure modes

- Mixing legacy exporter advice with the Dynamo/export path without version pinning.
- Giving dynamic-axis names while omitting valid range/shape policy.
- Assuming ONNX checker validates runtime kernel support or numerical parity.
- Dropping external weights when moving a model file.
- Registering a custom domain node without a runtime implementation/ABI plan.

## Mastery gate

Deliver a version-pinned ONNX artifact whose graph, inputs, dynamic contract, weights, and source provenance are inspectable. Demonstrate intended-runtime parity across allowed shape buckets and explain a deliberately unsupported operator's chosen boundary with its maintenance consequences.

## Primary sources, code paths, and OSS bridge

- [`torch.onnx` API](https://docs.pytorch.org/docs/stable/onnx.html)
- [PyTorch export-based ONNX exporter](https://docs.pytorch.org/docs/stable/onnx_export.html)
- [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX operator schemas](https://onnx.ai/onnx/operators/)

Read `torch/onnx/`, `test/onnx/`, [ONNX](https://github.com/onnx/onnx), and exporter reports at the exact versions under test. Upstream changes need an export + checker + intended-backend test, not a graph screenshot alone.

## Next

Module 53 executes the same artifact in ONNX Runtime and makes execution-provider partitioning visible.
