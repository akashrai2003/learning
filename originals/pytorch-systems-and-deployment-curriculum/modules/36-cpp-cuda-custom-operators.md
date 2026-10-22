# Module 36 — C++/CUDA Custom Operators

**Phase:** 6 — Native PyTorch and GPU integration  
**Time:** 15 focused hours  
**Prerequisites:** Modules 34–35, C++17, CUDA execution/streams, and build basics.

## Existing inference curriculum bridge

This deepens [36 — CUDA execution, memory, and streams](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), [39 — Triton kernels](../../inference-engineering-curriculum/modules/39-triton-kernels.md), [40 — CUTLASS/CuTe](../../inference-engineering-curriculum/modules/40-cutlass-and-cute.md), [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), and [43 — Build/JIT/Ninja/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md).

## Why this matters and outcomes

Native operators are where framework semantics meet CUDA. A correct kernel must honor tensors’ device, dtype, shape, strides, stream, allocator lifetime, autograd, FakeTensor, and error contracts—not just produce right values after a global synchronize. This module turns a small reference/Triton operation into a production-shaped C++/CUDA extension without prematurely committing to an upstream PyTorch core change.

You will be able to design a C++/CUDA operator boundary, write CPU/CUDA registrations, launch on the current stream with device guarding, check errors asynchronously correctly, test dispatcher/autograd/FakeTensor behavior, and package a minimal reproducible extension.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P36 — C++/CUDA custom operators**.

1. [Custom C++ and CUDA operators tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — learn the API or system contract.
2. [C++ extension docs](https://docs.pytorch.org/docs/stable/cpp_extension.html) — study the underlying design, paper, or specification.
3. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch custom-ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch extension-cpp examples](https://github.com/pytorch/extension-cpp) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

The layers are:

```text
Python call → operator schema/dispatcher → C++ wrapper → CUDA launch
            → current stream → allocator-managed tensors → later consumer
```

A CUDA wrapper receives `at::Tensor`/`torch::Tensor` values and must validate its own capability contract: CUDA device, supported scalar types, rank/shapes, layout/stride requirement, and possibly alignment. Do not silently call `.contiguous()` inside a purported fast op without reporting/costing the copy; either support strides or expose a documented fallback. Use a device guard before touching device-specific state and obtain PyTorch’s **current** CUDA stream rather than launching on an assumed default stream. A later operation may be queued on that same stream, so a host synchronization inside the op destroys composability and hides races.

CUDA launches are asynchronous. Check launch errors close to the launch using the project’s supported macro/function, but reserve device synchronization for test/debug timing. Allocate outputs through PyTorch APIs so they participate in the caching allocator and stream lifetime rules; raw allocations and untracked streams easily create use-after-free or accidental global sync. If the wrapper creates temporaries consumed on another stream, follow the allocator’s stream-recording conventions rather than assuming a tensor’s storage is immortal.

Registration should reflect semantics. `TORCH_LIBRARY` declares a schema; `TORCH_LIBRARY_IMPL` registers a backend implementation. C++ registrations can be supplemented with Python `torch.library.register_fake` and autograd registrations where appropriate. For a differentiable op, prefer a composition of existing differentiable ATen ops if it is fast enough; otherwise implement/validate backward deliberately. For inference-only operations, explicitly reject `requires_grad` if there is no valid autograd path rather than returning silently wrong gradients.

## Implementation ladder in `torch-lab/`

1. **Reference:** keep `torch-lab/native/reference_rmsnorm.py` as a fully tested PyTorch composition. Write `op_contract.md` before C++: formula, supported shapes/dtypes/strides, aliases/mutation, stream, output precision, error behavior, and fallback.
2. **Optimized:** implement `torch-lab/native/csrc/rmsnorm.cpp` and `rmsnorm_cuda.cu`, plus CPU/CUDA registrations. Start with a simple correct CUDA kernel and current-stream/device guard. Add a separate explicit capability dispatch rather than an implicit “fast” path.
3. **Integrated:** expose it to Python, register fake/autograd as necessary, run `opcheck`, eager/compiled tests, multi-stream ordering tests, and compare against the Triton/Inductor/vendor baseline. Package a clean source build recipe; Module 40 addresses JIT/CMake/Ninja details.

## Three focused sessions

1. **Boundary design:** define the schema and reference/property tests, then write C++ validation/CPU fallback without CUDA optimization.
2. **CUDA correctness:** add a masked, current-stream kernel; test tail/noncontiguous/rejected cases, error reporting, and a producer→op→consumer multi-stream sequence.
3. **Framework integration:** add fake/autograd/compile tests, package the extension, profile versus baseline, and document why it should or should not become an upstream candidate.

## Experiments and deliverables

Commit `csrc/`, `CMakeLists.txt` or extension build config, `op_contract.md`, `test_native_op.py`, `test_stream_order.py`, `test_native_compile.py`, and `native_benchmark.json`. Capture source/flag hashes, compiler/CUDA/PyTorch/ABI/GPU/SM data, dispatch table, all support cases, launch config, reference errors, build/cold/warm time, memory, and profiler traces.

## Common failure modes

- Launching on the legacy/default stream rather than PyTorch’s current stream.
- Synchronizing in the operator so tests pass but graph/overlap performance collapses.
- Checking only contiguous FP16 tensors and calling unsupported inputs a kernel bug.
- Registering CUDA code but omitting FakeTensor, autograd, or CPU behavior expected by callers.
- Smuggling a C++ exception or CUDA error far from the operation that caused it.

## Mastery gate

Deliver a source-buildable operator with a published contract, correct CUDA/CPU dispatch or explicit error, current-stream ordering test, edge/property tests, FakeTensor/compile coverage, and fair profiler comparison. Explain one situation where the custom kernel must safely defer to PyTorch/Inductor instead of forcing execution.

## Primary sources, code paths, and OSS bridge

- [PyTorch C++/CUDA custom-op tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [C++ extension documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [PyTorch custom-ops guide](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [PyTorch source](https://github.com/pytorch/pytorch)

Read `aten/src/ATen/cuda/`, `torch/csrc/cuda/`, `torch/library.h`, `aten/src/ATen/native/`, and extension examples/tests. First upstream contributions should be an exact correctness/stream/fake test or an isolated bug fix with CPU/CUDA coverage. Follow PyTorch’s contributor instructions and let maintainers decide whether an operator belongs in core versus an out-of-tree extension.

## Next

Module 37 explains how the dispatcher selects the implementation you just registered.
