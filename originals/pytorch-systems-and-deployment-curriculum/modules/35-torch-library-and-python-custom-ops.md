# Module 35 — `torch.library` and Python Custom Operators

**Phase:** 5 — Capture, export, and compiler foundations  
**Time:** 12 focused hours  
**Prerequisites:** Modules 29–34, autograd/aliasing knowledge, and a tested reference implementation.

## Existing inference curriculum bridge

Use [28 — Quantized kernel/system integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [39 — Triton kernels](../../inference-engineering-curriculum/modules/39-triton-kernels.md), [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), and [43 — Build/JIT/Ninja/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md).

## Why this matters and outcomes

A custom kernel becomes a PyTorch operator only after it declares its semantic contract. `torch.library` is the preferred Python-level route for defining/implementing an out-of-tree op and making it visible to eager execution, FakeTensor, autograd, `torch.compile`, and export. It is also the fastest route to discovering that an apparently working kernel has incomplete metadata.

You will define schemas honestly, register device and fake implementations, add autograd where appropriate, use `opcheck` plus `gradcheck`, and decide whether a custom operator should be opaque or expressed as decomposable PyTorch operations.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P35 — torch.library and Python custom ops**.

1. [torch.library](https://docs.pytorch.org/docs/stable/library.html) — learn the API or system contract.
2. [Custom Python operators tutorial](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html) — study the underlying design, paper, or specification.
3. [Custom ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch custom-ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch extension-cpp examples](https://github.com/pytorch/extension-cpp) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

An operator name such as `torch.ops.torch_lab.fused_bias_silu` identifies a schema and registrations. The schema is a promise about arguments, outputs, mutation, and aliasing—not comments. For a pure operation `(Tensor x, Tensor bias) -> Tensor`, outputs must be new storage unless the schema says otherwise. In-place and `out=` patterns have specific supported contracts. Arbitrary aliases are difficult for functionalization and transforms, so do not hide them behind an unqualified `Tensor -> Tensor` declaration.

Register a real implementation for each supported device and a FakeTensor/meta implementation that returns tensors with correct shape, dtype, device semantics, and strides without reading data pointers. Fake execution permits compiler/export shape analysis. If the op is differentiable, register an autograd formula using saved context/inputs or use an established C++ autograd path; validate derivatives separately. `torch.library.opcheck` tests schema, fake behavior, autograd registration, and transform integration for representative inputs; it does not prove numerical math. Pair it with reference parity and `gradcheck` where training is supported.

An opaque custom op stops a compiler from looking inside its implementation, which can be correct for a library/Triton/CUDA kernel but forfeits fusion across the boundary. A composition of public PyTorch operators may be more optimizable and exportable. Choose based on measured benefit and semantic needs. Do **not** use `torch.compiler.allow_in_graph` to black-box ordinary Python: it bypasses frontend safety checks and the PyTorch documentation recommends a proper custom op instead.

API details evolve, particularly decorator versus registration styles. Pin a release and use its `torch.library` docs/source. Make import/registration idempotence and package lifetime explicit in a real project; an operator can disappear if the library object/extension registration is incorrectly scoped.

## Implementation ladder in `torch-lab/`

1. **Reference:** keep `torch-lab/ops/reference_bias_silu.py` as a plain PyTorch composition, including input validation and a documented mathematical/alias contract.
2. **Optimized:** create `torch-lab/ops/custom_bias_silu.py` using `torch.library.custom_op` or the pinned equivalent. Register CPU/CUDA implementations, `register_fake`, and an autograd formula if applicable. Run `torch.library.opcheck` on CPU/CUDA, contiguous/noncontiguous/tail shapes, and requires-grad cases.
3. **Integrated:** bind the Triton kernel from Module 34 behind capability checks and a pure-PyTorch fallback. Compile/export a caller, compare eager/compiled forward+backward, and inspect whether opacity helps or blocks surrounding fusion.

## Three focused sessions

1. **Contract first:** write schema, shape/dtype/device matrix, alias/mutation table, error behavior, and a reference before registering anything.
2. **Transform compatibility:** add fake/autograd registrations and `opcheck`; deliberately give wrong shape/alias metadata to understand the tests, then repair it.
3. **System decision:** compare decomposed and opaque variants in eager/compiled/export paths, with a profiler and parity suite. Make a reasoned boundary decision.

## Experiments and deliverables

Commit `op_contract.md`, `custom_bias_silu.py`, `test_opcheck.py`, `test_op_grad.py`, `test_op_compile.py`, `test_op_export.py`, and a dispatch/capability table. Record schema string, registrations, version, supported inputs, fake metadata, autograd formula, reference tolerance, compile/export outcomes, and failure messages.

## Common failure modes

- Declaring a mutation-free schema for an implementation that writes input storage.
- Returning an input/view where the schema promises fresh output.
- Writing a fake kernel that accesses `data_ptr()` or produces wrong strides/device.
- Treating `opcheck` success as proof of correct numerical output.
- Using `allow_in_graph` rather than registering the actual operation.

## Mastery gate

Deliver one custom op with explicit contract, reference parity, `opcheck`, gradient checks if differentiable, fake support, eager/compiled behavior, and a graceful unsupported-input fallback. Explain whether an opaque boundary is justified by measured systems evidence.

## Primary sources, code paths, and OSS bridge

- [`torch.library` API and `opcheck`](https://docs.pytorch.org/docs/stable/library.html)
- [PyTorch custom-ops landing guide](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [Extending PyTorch note](https://docs.pytorch.org/docs/stable/notes/extending.html)
- [`torch.compiler.allow_in_graph` guidance](https://docs.pytorch.org/docs/stable/generated/torch.compiler.allow_in_graph.html)
- [PyTorch custom C++/CUDA operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)

Read `torch/_library/`, `torch/library.py`, `torch/_custom_ops.py`, and custom-op tests in [PyTorch](https://github.com/pytorch/pytorch). Useful contributions are schema/fake/autograd tests, clean minimal transform failures, documentation, and capability diagnostics. Correctness contracts are contribution-worthy in themselves; performance is considered only after them.

## Next

Module 36 moves the same operator contract across the C++/CUDA extension boundary.
