# Module 51 — `torch.export` and Program Capture

**Phase:** 8 — Export and deployment  
**Time:** 15 focused hours  
**Prerequisites:** Modules 29–35, 42, and a tested eager `nn.Module`.

## Existing inference curriculum bridge

Use this with [17 — Modern model architecture forensics](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md), [28 — Quantized kernel and system integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), and [42 — Compiler stack: Dynamo, Inductor, and Triton](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md). Export is where an architecture's Python behavior must become an explicit artifact contract.

## Why this matters and outcomes

`torch.export` is not a generic model-saving button. It captures a graph of PyTorch operations plus assumptions about values, shapes, mutation, aliases, parameters, buffers, and control flow. A graph that runs on one example can still be unsound for the deployment inputs that matter. Treating constraints as first-class makes later ONNX, AOTInductor, TensorRT, and ExecuTorch failures much easier to localize.

You will be able to make an eager model exportable, declare and test dynamic-shape ranges, inspect an `ExportedProgram`, debug capture failures with a minimal reproducer, and maintain a parity/constraint test suite across PyTorch versions.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P51 — torch.export and program capture**.

1. [torch.export](https://docs.pytorch.org/docs/stable/export.html) — learn the API or system contract.
2. [Export programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/programming_model.html) — study the underlying design, paper, or specification.
3. [Export tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Capture begins with representative example inputs, but it produces more than a trace. The graph signature identifies user inputs, parameters, buffers, and outputs; range constraints define allowable symbolic dimensions; guards preserve assumptions. Non-Tensor inputs are normally specialized at export time, so passing `True`, a Python integer, or a mode string may silently make a different artifact rather than a flexible input. Tensor state shapes are generally fixed at export time unless represented through supported dynamic inputs; read your installed version's programming model before relying on any detail.

Current PyTorch documentation describes strict and non-strict capture modes. Strict capture uses TorchDynamo for stronger Python-level safety checks; the current programming-model page recommends non-strict for many models and indicates defaults are evolving. Do not hard-code that recommendation into an unpinned deployment script: record `torch.__version__`, mode, inputs, constraint declarations, and generated artifact. The core contract is that inputs satisfying captured assumptions should execute equivalently; inputs outside them should be rejected or handled by a separately designed artifact, not accidentally accepted.

Export legality is related to but not identical to `torch.compile`. Python data-dependent control flow, unsupported mutation/aliasing, custom Python objects, dynamic output structure, and unregistered custom ops are frequent blockers. The repair order is: expose the true interface, write eager tests, isolate the smallest failing operation, choose an allowed PyTorch expression or a properly registered op, and revalidate both semantics and constraints. Avoid “fixing” a failure by declaring all shapes static unless that is the actual deployment contract.

## Implementation ladder in `torch-lab/`

1. **Reference:** build `torch-lab/export/eager_contract.py` around the compact decoder or vision model. Its tests cover valid shapes, boundary shapes, invalid shapes, state/buffer behavior, output structure, and numeric tolerances.
2. **Optimized:** add `export_program.py` that exports one fixed-shape and one deliberately bounded dynamic-shape artifact. Save graph code/signature/range constraints and an input-contract JSON alongside the artifact.
3. **Integrated:** write `export_failure_notebook.md` with three minimal failures—Python data branch, mutation/alias issue, and custom-op issue—and a principled resolution or an explicit non-exportable boundary. Test `ExportedProgram.module()` on every allowed shape bucket.

## Three focused sessions

1. **Capture anatomy:** export a small module, inspect graph/signature/constraints, and explain what becomes a user input, parameter, buffer, constant, or specialized value.
2. **Dynamic contract:** declare a batch/sequence range, generate in-range and out-of-range cases, and test both numerical parity and expected rejection behavior.
3. **Failure reduction:** reduce a failed real model export to the smallest module and input. Document the original source location, failure category, attempted fixes, chosen contract, and regression test.

## Experiments and deliverables

Commit `eager_contract.py`, `export_program.py`, `test_export_contract.py`, `artifact_manifest.json`, `graph_dumps/`, and `export_failure_notebook.md`. The manifest records PyTorch/python version, exact mode, model/state hash, input pytree/schema, dynamic dimensions and bounds, graph signature, ops used, tolerance, valid/invalid test results, artifact hash, and known unsupported inputs.

## Common failure modes

- Assuming example-input success means arbitrary dynamic inputs are valid.
- Allowing a Python scalar or flag to specialize silently without documenting it.
- Treating the graph's successful execution as parity evidence without boundary tests.
- Keeping a data-dependent Python branch outside the declared artifact contract.
- Mutating a buffer/input and ignoring the exported graph signature's mutation semantics.

## Mastery gate

Ship an `ExportedProgram` plus a human-readable input/state contract and an automated allowed/out-of-contract test matrix. From a capture error, isolate the smallest failing program and explain whether the correct fix is code restructuring, a registered operator, a separate artifact, or a deliberate non-export decision.

## Primary sources, code paths, and OSS bridge

- [`torch.export` programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/programming_model.html)
- [`torch.export` API](https://docs.pytorch.org/docs/stable/export.html)
- [Export challenges and solutions](https://docs.pytorch.org/tutorials/recipes/torch_export_challenges_solutions.html)
- [PyTorch export source](https://github.com/pytorch/pytorch/tree/main/torch/export)

Read `torch/export/`, `torch/_export/`, and export tests matching your pinned release. A useful upstream report includes the smallest executable model, inputs, expected contract, full version data, and no proprietary checkpoint; a fix needs parity and constraint regression tests.

## Next

Module 52 lowers a declared export contract into ONNX IR rather than treating ONNX as a black-box file conversion.
