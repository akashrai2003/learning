# Module 06 — Framework Landscape and Interoperability

> **Phase:** I — Engineering foundations  
> **Time:** Week 3; three focused 3-hour sessions  
> **Prerequisites:** Modules 01–05 and a small tested PyTorch model

## Existing inference curriculum bridge

Use this map alongside [Module 17 — `modern-model-architecture-forensics.md`](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md), [Module 28 — `quantized-kernel-and-system-integration.md`](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [Module 41 — `pytorch-custom-operators.md`](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [Module 42 — `compiler-stack-dynamo-inductor-and-triton.md`](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), Module 46 — `vllm-internals-and-contributions.md`, Module 47 — `sglang-internals-and-contributions.md`, and Module 48 — `llamacpp-ggml-internals-and-contributions.md`. The point is to move models and semantics across boundaries deliberately—not to collect logos.

## Why this matters and outcomes

PyTorch is the home base: it exposes eager semantics, autograd, dispatcher/custom-op interfaces, compilation, export, and a broad research ecosystem. Become deep enough to read and contribute to it. Study JAX/XLA for functional transformations, TensorFlow for graph/runtime context, ONNX for interchange, TensorRT for NVIDIA engine compilation, ExecuTorch for edge targets, and `torchtune` for PyTorch-native post-training.

You will be able to draw the boundary between a model’s mathematical semantics, an executable graph/IR, weights, custom operators, and a target runtime; select an interchange path from evidence; and build parity tests that catch a valid-looking but behaviorally different exported model.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P06 — Framework landscape and interoperability**.

1. [torch.export](https://docs.pytorch.org/docs/stable/export.html) — learn the API or system contract.
2. [ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html) — study the underlying design, paper, or specification.
3. [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: interoperability transports contracts, not magic

An artifact is a layered object:

```text
architecture/config + parameter tensors + tokenizer/pre/post-processing
  + graph/IR + operator schemas + quantization/layout metadata + runtime policy
```

Moving only the graph rarely moves the model. Dynamic control flow, custom ops, mutable KV cache, sampling, tokenizer normalization, RoPE, dtype casts, and quantization packing remain semantic obligations. Export succeeds only over its declared shape/dtype/device range.

Different systems answer different questions:

| System | Study it for | Do not infer |
|---|---|---|
| PyTorch | eager semantics, autograd, dispatcher, compile/export | that eager Python is a deployment artifact |
| JAX | `jit`/`vmap`/`grad`, functional state | that it replaces PyTorch serving internals |
| TensorFlow | graph/runtime context | that API familiarity is PyTorch depth |
| ONNX | operator-graph interchange | that every PyTorch op/layout is portable |
| TensorRT / Torch-TensorRT | NVIDIA engines, precision, profiles | that arbitrary dynamic shapes/plugins work |
| ExecuTorch | edge runtime, memory planning, delegates | that server CUDA work transfers to edge |

## Theory and graph-boundary contracts

Eager PyTorch executes immediately; `torch.compile` captures eligible guarded regions; `torch.export` produces an ahead-of-time graph under explicit input assumptions. These are distinct products. Unsupported ops, data-dependent control flow, aliases/mutations, and dynamic-shape relations must be represented or refactored.

Graph portability has a set-intersection constraint. If source graph operators are \(S\), target-supported operators are \(T\), and custom/plugin lowering rules are \(C\), portable execution needs each semantic operation in \(S\) mapped into \(T\cup C\) with equivalent type, shape, state, and numerical behavior. A mere successful serialization proves syntax; parity testing proves a limited semantic domain.

For a decoder, define the boundary intentionally. Full generation includes host sampling and growing state; a deployable unit might be `prefill(...) -> logits, kv_state` and `decode(...) -> logits, kv_state'`. This may not match paged KV representation. A target integration can delegate only a stateless encoder, GEMM, or bounded-shape subgraph.

Zero-copy interchange is also conditional. DLPack transfers tensor ownership/metadata between supporting libraries; it does not solve device synchronization, dtype/layout convention, or autograd ownership. Treat it as a narrow, tested bridge rather than an excuse to pass undocumented raw pointers.

## Implementation ladder in the proposed `torch-lab/`

1. **Reference:** make the toy classifier from Module 01 a pure PyTorch baseline with a portable `forward(x)` and a fixed inference fixture. Write an artifact manifest covering weights, preprocessing, classes, input ranges, and expected logits.
2. **Optimized:** export a stateless `torch.export` path, then ONNX where supported. Compare target outputs over fixed and edge-shape fixtures; record non-representable operations rather than substituting them.
3. **Integrated:** create `deploy/` adapters for Torch-TensorRT and ExecuTorch. Keep eager PyTorch as oracle; gate artifacts on parity, shape-profile coverage, size, warm/cold latency, and diagnostics.

## Three focused sessions

1. **Session A — map the boundaries (3h):** inventory a model’s Python, ATen ops, state, preprocessing, and custom kernels. Mark weights, graph, config, or runtime policy; read `torch.export` constraints.
2. **Session B — one honest export (3h):** export a small stateless model, inspect its graph, execute it, and create shape/dtype parity tests. Add a data-dependent Python branch to see why eager success is not exportability.
3. **Session C — compare target economics (3h):** profile eager PyTorch versus one compiled/exported path using the same fixtures. Report artifact size, build time, warm latency, supported dynamic range, and debugging cost—not a single speed number.

## Experiments and deliverables

Write a decision record: source contract, target runtime, operator coverage, dynamic shapes, precision, custom/plugin story, numerical gate, and rollback. Add DLPack only where two installed libraries support it; test ownership and stream ordering. Start a `torchtune` tiny-model experiment and retain its recipe/config and conversion assumptions.

## Common failure modes

- Exporting a model but omitting tokenizer, normalization, position IDs, output decoding, or cache semantics.
- Treating ONNX operator support as a guarantee of exact low-precision or dynamic-shape behavior.
- Tuning TensorRT precision/profiles against a single friendly input and then calling the engine general-purpose.
- Solving an unsupported custom operator by replacing it with a mathematically different eager fallback without updating the parity gate.
- Learning JAX or TensorFlow API syntax before being able to inspect a PyTorch eager/ATen/dispatcher path.

## Mastery gate

Choose one bounded model function, produce a PyTorch oracle and one exported/target artifact, and demonstrate parity across a stated input domain including an odd shape. Explain every non-portable feature and whether it is graph, operator, state, or runtime policy. Write a one-page decision record that justifies why PyTorch remains the authoring/debugging source of truth.

## Primary sources and canonical code paths

- [PyTorch export documentation](https://docs.pytorch.org/docs/stable/export.html), [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html), [DLPack APIs](https://docs.pytorch.org/docs/stable/dlpack.html), and [`torch/export`](https://github.com/pytorch/pytorch/tree/main/torch/export).
- [Torch-TensorRT documentation](https://pytorch.org/TensorRT/), [NVIDIA TensorRT developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html), and [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/).
- [JAX documentation](https://docs.jax.dev/), [TensorFlow guide](https://www.tensorflow.org/guide), [ONNX specification](https://onnx.ai/onnx/), and [torchtune documentation](https://docs.pytorch.org/torchtune/stable/).
- Inspect [`pytorch/pytorch/torch/onnx`](https://github.com/pytorch/pytorch/tree/main/torch/onnx), [`pytorch/executorch`](https://github.com/pytorch/executorch), [`pytorch/TensorRT`](https://github.com/pytorch/TensorRT), and [`pytorch/torchtune`](https://github.com/pytorch/torchtune).

## OSS bridge and next

Interoperability bugs are high-value contributions when they include a small exported graph, operator/schema details, expected/actual tensors, and a target version. PyTorch depth stays the priority: spend most implementation time in its eager, compiler, and dispatcher layers; use other frameworks as contrast and deployment constraints. Next, understand PyTorch’s fundamental storage model in [Module 07](07-tensor-storage-views-strides-and-layouts.md).
