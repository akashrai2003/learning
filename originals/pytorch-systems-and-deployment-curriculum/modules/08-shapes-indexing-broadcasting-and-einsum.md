# Module 08 — Shapes, Indexing, Broadcasting, and einsum

> **Phase:** I — Engineering foundations  
> **Time:** 8–10 focused hours across three sessions  
> **Prerequisites:** Modules 01–07, basic linear algebra, and Python sequence/indexing fluency

## Existing inference curriculum bridge

Implement the contracts from [Module 03 — Tensor Math, Shapes, and einsum](../../inference-engineering-curriculum/modules/03-tensor-math-shapes-and-einsum.md) in PyTorch. Revisit [Module 06 — Transformer Forward Pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md) once you can reason about every reshape and allocation it hides. The inference curriculum gives a framework-light oracle; this module makes PyTorch tensor semantics a tool rather than a convenience.

## Why this matters and outcomes

Most model and kernel bugs are not arithmetic mistakes. They are semantic shape mistakes that happen to return a tensor: a head axis is swapped with time, a broadcast becomes [B,S,S], or a harmless-looking reshape silently copies. PyTorch makes these expressions concise, so a written tensor contract matters even more.

By the end, you can derive an operation’s output axes, reduction axes, strides, and likely allocation before running it; distinguish views from copies and broadcasted views; use basic and advanced indexing deliberately; express a contraction as loops, einsum, and matmul/bmm; and write adversarial tests that expose mistakes obscured by B=1 or square toy shapes.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P08 — Shapes, indexing, broadcasting, and einsum**.

1. [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) — learn the API or system contract.
2. [torch.einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) — study the underlying design, paper, or specification.
3. [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ATen native source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

A dense tensor has dtype, device, shape, strides, storage offset, and sometimes a memory format. For a contiguous row-major [a,b,c] tensor, strides are normally [b*c,c,1], so x[i,j,k] lives at storage_offset + i*stride[0] + j*stride[1] + k*stride[2]. Shape names logical axes; strides describe storage address arithmetic. A transpose or permute normally changes metadata, not bytes. A later operation can still require contiguous input and make a copy.

View is a strict storage reinterpretation with compatible strides. Reshape is a semantic request which may return a view or allocate a copy. Check rather than assume: inspect shape, stride, storage_offset, is_contiguous, and storage identity. Basic indexing with slices, integers, ellipsis, and None usually produces views. Advanced indexing with long or boolean index tensors generally gathers a new result; assignment through indexing has special in-place behavior. The correct habit is to prove an aliasing claim with a mutation test, not repeat a rule from memory.

Broadcasting lines dimensions up from the right. Each aligned pair must be equal, one, or absent. Thus [B,H,S,D] multiplied by [H,1,D] means that scale is shared over B and S. Expand creates a zero-stride logical view, excellent for read-only broadcasting; repeat copies data. Broadcasting may avoid an input copy while the consuming pointwise operation still creates a full output.

Einstein notation makes contractions explicit:

Y[b,s,f] = sum_d X[b,s,d] W[d,f].

The PyTorch expression is einsum("bsd,df->bsf", X, W). Indices absent from output are reduced. Attention scores are einsum("bhqd,bhkd->bhqk", Q, K), logically producing the [B,H,Sq,Sk] quadratic matrix. Production implementations often flatten [B,S,D] to [B*S,D] for GEMM or choose bmm after arranging batch axes. Equal mathematics does not imply equal layout, temporary memory, or selected kernel.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/foundations/tensor_contracts.py. Every helper must state axis order, dtype/device, output shape, accepted layout, and aliasing/mutation behavior.
2. Implement linear, split_heads, merge_heads, causal scores, and value aggregation with loops on tiny CPU tensors.

**Optimized**

3. Add einsum, matmul, and bmm forms, then compare against the loop oracle using asymmetric dimensions: B=2, H=3, S=5, Dh=7.
4. Add layout_report(t) that records shape, stride, storage offset, element size, storage identity, and contiguity. Demonstrate permute, contiguous, view, reshape, expand, repeat, and each indexing category.

**Integrated**

5. Add property tests: inverse permutations, identity weights, loop/einsum/GEMM parity, and preservation under non-contiguous inputs where the API contract promises it.

Use an axis legend everywhere: B active sequences, S tokens, H query heads, Hkv KV heads, Dh head width, D model width, F FFN width, and V vocabulary. Never allow equal dimensions to substitute for a semantic contract.

## Three focused sessions

1. **Metadata and views (3h):** construct contiguous and non-contiguous tensors, manually calculate addresses, and prove which transformations share storage. Compare expand and repeat, including a deliberate invalid in-place write to a zero-stride view.
2. **Indexing and broadcasting (3h):** predict output shapes before execution. Build a table for basic indexing, advanced indexing, masks, gather, index_select, and assignment. Produce one broadcast bug that passes at B=1 and fails at B>1.
3. **Contractions (3h):** derive QKV projection and attention equations, implement each three ways, then benchmark contiguous and transposed inputs only after the correct CUDA synchronization boundary.

## Experiments and deliverables

Commit the contract library, property tests, an einsum worksheet, and a layout report. For six operations, give input/output axes, FLOPs, strides, expected aliases, and expected allocations. Add a one-page source walk of a real decoder’s QKV reshape, including why its exact layout was chosen.

## Common failure modes

- Calling view after transpose as if it always reinterprets storage.
- Using repeat where expand expresses a read-only broadcast.
- Treating advanced indexing as a view.
- Testing only singleton, square, or equal-sized axes.
- Assuming a broadcast means the whole enclosing expression is allocation-free.
- Timing asynchronous CUDA operations without an intentional synchronization policy.

## Mastery gate

Given an unfamiliar contraction, state axes, output shape, reduced indices, approximate FLOPs, likely lowering, and quadratic intermediates. Given shape/stride/offset data, predict whether a reshape can be a view. All reference implementations must pass against asymmetric and non-contiguous cases.

## Primary sources and canonical code paths

- [PyTorch tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [PyTorch broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
- [PyTorch indexing, slicing, joining, and mutating operations](https://docs.pytorch.org/docs/stable/torch.html#indexing-slicing-joining-mutating-ops)
- Inspect [TensorIndexing.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIndexing.h), [torch/_refs](https://github.com/pytorch/pytorch/tree/main/torch/_refs), and one production model implementation.

## OSS bridge and next

A good first PyTorch contribution here is a narrowly scoped indexing/view regression test with an asymmetric minimal reproducer. Read the existing test next to the behavior before proposing an implementation change. Module 09 turns this metadata literacy into systems literacy: devices, dtypes, allocator behavior, and streams.
