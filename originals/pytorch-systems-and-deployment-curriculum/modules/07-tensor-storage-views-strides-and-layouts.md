# Module 07 — Tensor Storage, Views, Strides, and Layouts

> **Phase:** II — PyTorch tensor core  
> **Time:** Week 4; three focused 3-hour sessions  
> **Prerequisites:** Modules 01–05 and the tensor contracts in Module 04

## Existing inference curriculum bridge

This module grounds [Module 03 — `tensor-math-shapes-and-einsum.md`](../../inference-engineering-curriculum/modules/03-tensor-math-shapes-and-einsum.md), [Module 28 — `quantized-kernel-and-system-integration.md`](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [Module 36 — `cuda-execution-memory-and-streams.md`](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), [Module 37 — `gemm-and-tensor-cores.md`](../../inference-engineering-curriculum/modules/37-gemm-and-tensor-cores.md), and [Module 40 — `cutlass-and-cute.md`](../../inference-engineering-curriculum/modules/40-cutlass-and-cute.md). “Layout” is the bridge between correct Python tensor notation and the addresses a kernel consumes.

## Why this matters and outcomes

Many GPU-performance mysteries are materialized copies: a cheap `permute` is followed by a costly `reshape`, then a hot loop pays every token. Code assuming contiguity can reject legal inputs or corrupt overlapping views. See a tensor as storage plus metadata, not a nested array.

You will be able to compute addresses from shape/stride/offset, predict which operations return views versus copies, reason about aliasing and in-place behavior, inspect layout before profiling it, and specify kernel input-layout requirements without unnecessary `contiguous()` calls.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P07 — Tensor storage, views, strides, and layouts**.

1. [Tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html) — learn the API or system contract.
2. [torch.as_strided](https://docs.pytorch.org/docs/stable/generated/torch.as_strided.html) — study the underlying design, paper, or specification.
3. [PyTorch tensor semantics](https://docs.pytorch.org/docs/stable/tensors.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ATen native source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: one storage, many coordinate systems

For a dense strided tensor, the important metadata is:

```text
untyped storage + dtype + device + sizes + strides + storage_offset + layout/memory format
```

If element width is \(b\) bytes and index vector is \(i=(i_0,\ldots,i_{n-1})\), the element address is

\[
\text{byte address}=\text{base}+b\left(\text{storage\_offset}+\sum_k i_k\,\text{stride}_k\right).
\]

For `x = torch.empty(2, 3, 4)`, conventional contiguous strides are `(12, 4, 1)`: incrementing the final dimension moves one element; incrementing batch skips twelve. `x.transpose(1, 2)` has shape `(2,4,3)` and strides `(12,1,4)` while sharing storage. It changes interpretation, not bytes.

“Contiguous” is a layout predicate, not validity. A transposed tensor is valid and may be efficiently consumed by a stride-aware kernel. If a kernel requires contiguity/alignment or packing, select another kernel or materialize once at a deliberate boundary.

## Theory: views, copies, aliases, and formats

Basic slicing/indexing, `select`, `narrow`, `transpose`, `permute`, and `expand` commonly yield sharing views. Advanced integer/boolean indexing generally copies; assignment through either style writes into the original. Confirm with `untyped_storage().data_ptr()`, `storage_offset()`, `stride()`, and profiling.

`view(new_shape)` succeeds only when existing strides represent the requested dimensions; it never copies. `reshape`/`flatten` may view or allocate, so code must not depend on which. `contiguous()` returns self or copies; `clone()` creates independent storage. `.to()` may be a no-op or allocation.

`expand` represents a size-one dimension with stride zero: `scale:[D] -> [B,T,D]` reuses each `scale[d]`. In-place writes to expanded/overlapping views are invalid or undefined. `as_strided` exposes arbitrary size/stride/offset but can create overlaps or out-of-bounds layouts; prefer purpose-built views.

Dense `torch.strided` is common. `channels_last`/`channels_last_3d` help some convolution paths, not generic LLM `[B,T,D]`. Sparse/nested forms differ. Quantized/KV/attention kernels may have opaque packed layouts; attach format/version metadata and dispatch only compatible kernels.

Autograd tracks view relationships and version counters so it can reject certain in-place modifications needed for backward. Do not silence an in-place error with `.data` or `detach()`; identify whether the mutation is semantically intended, move it under `no_grad` for optimizer state, or make an explicit copy. Alias behavior also matters to `torch.compile` and functionalization: hidden mutation shrinks what a compiler can safely reorder/fuse.

## Implementation ladder in the proposed `torch-lab/`

1. **Reference:** add `ops/layout_inspect.py` printing size, stride, offset, dtype, device, pointer identity, `is_contiguous`, and `is_contiguous(memory_format=...)`. Build hand-checked 2D/3D examples and a tiny address calculator that agrees with indexing.
2. **Optimized:** implement a layout-aware `split_heads`/`merge_heads` pair for `[B,T,D] ↔ [B,H,T,Dh]`. First use explicit `transpose` and `reshape`, then benchmark pre-materializing once versus allowing a downstream operation to materialize. Test every output against a contiguous reference.
3. **Integrated:** define layout preconditions on `torch_lab` operator interfaces. The attention module chooses a generic reference path for arbitrary strides and a fast path only after an explicit, measured conversion; logs the conversion bytes/time as part of the operation cost.

## Three focused sessions

1. **Session A — address arithmetic (3h):** create a 2×3×4 tensor with recognizable values; derive five indexed addresses by hand; validate pointer, offset, and stride metadata after slice/transpose/narrow.
2. **Session B — alias audit (3h):** compare `view`, `reshape`, `flatten`, `clone`, basic slicing, advanced indexing, and `expand`. Mutate legal views, detect shared storage, and write tests around expected alias/no-alias behavior. Construct an overlapping `as_strided` example only in an isolated test and never mutate it.
3. **Session C — performance boundary (3h):** benchmark a representative head layout through `matmul`/SDPA with contiguous and noncontiguous inputs. Use profiler memory data to locate copies; compare moving one conversion outside a decode loop with repeating it inside.

## Experiments and deliverables

Create a layout dossier containing before/after size-stride-offset tables, storage-pointer diagrams, and profiler traces for one unexpected copy. Add property tests that `merge_heads(split_heads(x)) == x` for odd valid shapes, while preserving/declaring output layout. Include a failed `view` test and explain why `reshape` could still work by copying.

## Common failure modes

- Calling `.contiguous()` defensively everywhere, converting a metadata issue into sustained bandwidth traffic.
- Assuming `reshape` cannot copy or that `permute` always causes a copy.
- Performing in-place updates through `expand` or overlapping `as_strided` views.
- Storing a packed quantized tensor as though ordinary `dtype`, shape, and strides describe its interpretation.
- Testing only contiguous inputs, then blaming a kernel when a legal PyTorch view hits an undocumented layout assumption.

## Mastery gate

Given an arbitrary sequence of `slice`, `transpose`, `reshape`, and `expand`, predict shape, strides, shared-storage status, and whether a requested view is legal before running it. Diagnose a profiler trace containing an implicit copy and redesign the interface so it happens at most once per request, with numerical parity and documented cost.

## Primary sources and canonical code paths

- [PyTorch tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html), [storage](https://docs.pytorch.org/docs/stable/storage.html), [tensor attributes/memory formats](https://docs.pytorch.org/docs/stable/tensor_attributes.html), [`view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html), and [`as_strided`](https://docs.pytorch.org/docs/stable/generated/torch.as_strided.html).
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) for view/in-place constraints.
- Inspect [`aten/src/ATen/TensorImpl.h`](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h), [`aten/src/ATen/native/TensorShape.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorShape.cpp), [`torch/_subclasses/functional_tensor.py`](https://github.com/pytorch/pytorch/blob/main/torch/_subclasses/functional_tensor.py), and view tests in [`test/test_view_ops.py`](https://github.com/pytorch/pytorch/blob/main/test/test_view_ops.py).

## OSS bridge and next

Layout bugs make excellent upstream contributions when accompanied by a small tensor printout of sizes, strides, offset, device, expected semantics, and a regression test. This storage literacy unlocks the shape/indexing language used everywhere next in [Module 08](08-shapes-indexing-broadcasting-and-einsum.md).
