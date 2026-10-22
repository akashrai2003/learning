# Module 41 — PyTorch Custom Operators

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 21, paired with Module 42; three 3-hour sessions and a packaged-operator review  
**Prerequisites:** Modules 35–40; CUDA streams, kernel correctness, Triton/CUTLASS implementations, and Python packaging basics

## Why this matters

A fast CUDA kernel is not yet a PyTorch operator. PyTorch needs a schema, dispatch registrations, shape/dtype metadata for FakeTensor, device and stream correctness, mutation/aliasing truth, and often autograd or autocast behavior. Missing metadata may appear to work eagerly and fail under `torch.compile`; a false alias declaration can let a compiler reorder writes and silently corrupt results.

## Outcomes

You can:

- define an operator schema that accurately states types, aliasing, and mutation;
- register CPU/reference, CUDA, FakeTensor, and optional autograd/autocast behavior;
- launch on PyTorch's current CUDA stream and check the active device;
- use `torch.library.opcheck` for registration and separate numerical tests for math;
- package a C++/CUDA extension and integrate it through eager and compiled execution.

## Mental model: one semantic operator, several dispatch implementations

The dispatcher selects an implementation from the operator name, overload, tensor dispatch keys, device, modes, and other state. A conceptual fused operation

\[
Y_{[M,N]}=\operatorname{SiLU}(X_{[M,K]}W_{g,[N,K]}^\top)
\odot(X_{[M,K]}W_{u,[N,K]}^\top)
\]

needs more than this equation. Its contract includes legal dtypes/devices, layout/stride requirements, output shape and stride, accumulation dtype, aliases, mutated arguments, determinism, zero/tail shapes, and errors for unsupported cases.

Register the schema once, implementations by dispatch key. In C++, `TORCH_LIBRARY` defines schemas and `TORCH_LIBRARY_IMPL` supplies backend kernels; Python's `torch.library` can define or extend operators. A FakeTensor implementation executes without data storage and returns only correct metadata. Dynamo and downstream compiler passes rely on it for shape propagation, so it must not read a data pointer or calculate values from tensor contents.

`opcheck` verifies registration properties: schema agreement, FakeTensor behavior, autograd registration, and AOT/dynamic compatibility. It does **not** establish numerical correctness. Compare outputs against a high-precision composition, fuzz shapes/strides, and use `gradcheck` if gradients are supported.

The CUDA binding must guard the device and use the framework's current stream. Launching on an assumed default stream can race with surrounding PyTorch work. Temporary tensors must obey allocator stream-lifetime rules. Report launch failures near the launch rather than letting an unrelated later operation surface them.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/custom_swiglu.py` as a composition of public PyTorch operations. Write the complete contract before native code and test CPU/CUDA, noncontiguous inputs, tails, empty dimensions, and rejected dtypes.

### 2. Optimized

Add `inference-lab/kernels/custom_op/` with a C++ schema/binding and the CUDA or Triton implementation from Modules 39–40. Register a Python FakeTensor kernel. Use an explicit fallback or a clear `TORCH_CHECK` for unsupported SM/shape combinations. Build native code for the local RTX 5060 Ti's `sm_120`, not a datacenter Blackwell target.

### 3. Integrated

Package the extension in `inference-lab/engine/ops/`, import it from a small module, and run eager plus `torch.compile(fullgraph=True)`. Test a non-default stream, CUDA graph capture if supported, autocast, serialization/import in a clean process, and several dynamic shapes. Confirm which backend ran with structured diagnostics.

## Three 3-hour sessions

1. **Schema and dispatcher:** specify semantics, register a CPU/reference implementation, inspect dispatch, and intentionally fail alias/mutation tests.
2. **CUDA and metadata:** bind the optimized kernel, honor device/current stream, register FakeTensor behavior, and run `opcheck` across shapes and devices.
3. **Packaging and compile:** build a wheel or editable package, test clean-process loading, run under Dynamo/Inductor, and profile the integrated call.

## Experiments and deliverables

Commit:

- `reference/custom_swiglu.py`;
- `kernels/custom_op/custom_swiglu.cpp`;
- `kernels/custom_op/custom_swiglu.cu`;
- `engine/ops/custom_swiglu.py`;
- `tests/test_custom_swiglu_opcheck.py`;
- `benchmarks/results/custom-op-eager-compile.json`;
- `notes/source_walks/pytorch-dispatcher.md`.

Record PyTorch and CUDA versions, compiler, C++ ABI setting, architecture flags, build command, schema, dispatch table, supported shape/dtype matrix, tolerance, current-stream test, cold import/build time, and warm kernel/model timing.

## Common failure modes

- Treating a pybind function as a fully composable dispatcher operator.
- Declaring an operation functional while mutating or aliasing input storage.
- Returning plausible FakeTensor shapes with wrong strides, dtype, or device.
- Accessing real data inside the fake implementation.
- Assuming `opcheck` proves mathematical accuracy.
- Forgetting current-device guards or launching on stream zero.
- Relying on a CUDA fallback that silently copies through CPU.
- Building only an `sm_100` cubin and calling it generic Blackwell support.
- Importing a `.so` built against an incompatible PyTorch/CUDA ABI.

## Mastery gate

From an empty package, implement and load one CPU/CUDA operator. Show its schema and dispatch table, pass `opcheck` on representative inputs, pass independent oracle tests including tails/noncontiguity, and run inside a full graph with no break. Demonstrate a cross-stream producer/consumer test and explain every ownership edge. A clean virtual environment must import the built artifact reproducibly.

## Canonical code areas and OSS bridge

Read PyTorch's dispatcher APIs around `torch/library.h`, Python registration under `torch/_library/`, extension machinery in `torch/utils/cpp_extension.py`, and native examples in `extension-cpp`. Then inspect custom-op wrappers and tests in vLLM's `_custom_ops.py`, `csrc/`, and `tests/kernels/`. A good first contribution is a missing fake implementation, `opcheck` case, schema correction, stream test, or capability diagnostic—not a large unreviewed kernel rewrite.

## Primary and authoritative sources

- [PyTorch custom C++/CUDA operator tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [`torch.library` and `opcheck`](https://docs.pytorch.org/docs/stable/library.html)
- [PyTorch dispatcher tutorial](https://pytorch.org/tutorials/advanced/dispatcher.html)
- [PyTorch C++ extension documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [PyTorch source](https://github.com/pytorch/pytorch)

## Next

Module 42 follows this operator through Dynamo graph capture, FakeTensor propagation, Inductor lowering, Triton code generation, guards, and caches.
