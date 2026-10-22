# Module 31 — Guards, Recompilation, and Dynamic Shapes

**Phase:** 5 — Capture, export, and compiler foundations  
**Time:** 9 focused hours  
**Prerequisites:** Modules 29–30, shape algebra, and a representative model workload.

## Existing inference curriculum bridge

Study alongside [12 — MQA/GQA and KV cache](../../inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md), [14 — Long context and memory](../../inference-engineering-curriculum/modules/14-long-context-and-memory.md), [31 — PagedAttention](../../inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md), [34 — Continuous batching](../../inference-engineering-curriculum/modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md), and [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md).

## Why this matters and outcomes

Shape variability is the normal case for serving: prompt lengths, batch size, image tokens, and decode steps change. Specialization can create excellent kernels; uncontrolled specialization causes recompile storms, cache bloat, latency spikes, and eventually eager fallback. Dynamic shapes are a proof/optimization tradeoff, not a checkbox.

You will understand what guards assert, observe guard failures and recompiles, choose static buckets versus symbolic ranges based on workload evidence, express export constraints, and test boundary shapes before deploying a compiled function.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P31 — Guards, recompilation, and dynamic shapes**.

1. [Dynamic shapes manual](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html) — learn the API or system contract.
2. [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) — study the underlying design, paper, or specification.
3. [Dynamo guards source](https://github.com/pytorch/pytorch/tree/main/torch/_dynamo) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

A compiled artifact is valid only if its assumptions still hold. Guards can concern Python globals/closures, module identity, tensor dtype/device/requires-grad, rank, sizes, strides/layout, and symbolic relations. Informally, a cache entry is \((G,\phi)\), where graph \(G\) may run only when guard predicate \(\phi(x)\) holds. A new input that violates \(\phi\) can trigger a new specialization, a more dynamic trace, or fallback depending on configuration and limits.

Symbolic dimensions let code reason about values like \(B,T\) without assigning one concrete integer, subject to constraints. The attention shape `(B,H,T,D)` still contains non-symbolic requirements: head dimension can select a kernel, `T` may need divisibility/padding, strides may determine vectorization, and memory allocations must fit. “Dynamic” does not mean every data-dependent result is universally supported; a value computed from data may be unbacked and require explicit handling or prevent an optimization.

In `torch.compile`, `dynamic=None` lets the compiler adapt after a shape-induced recompile; `dynamic=True` asks it to seek a dynamic kernel up front; `dynamic=False` specializes. Exact behavior is release/backend dependent, so inspect diagnostics. For deployment, `torch.export` has a stronger interface: describe allowed ranges with `torch.export.Dim` (and derived relations where supported), then validate constraints at runtime. Never declare an unbounded sequence dimension simply because a model theoretically accepts it; memory and algorithms have real limits.

Serving usually benefits from evidence-driven buckets: e.g., pad selected prompt lengths to a small set, group compatible requests, precompile/warm those shapes, and keep an eager/general path for rare tails. Padding trades extra FLOPs and KV bytes for fewer cold compilations. Compare it against truly dynamic kernels using p50/p99—not only mean throughput.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/compiler/shape_matrix.py` that executes a small attention/MLP block over a matrix of `(B,T,D,stride,dtype)` and verifies eager outputs/gradients. Include zero/one, tail, and noncontiguous cases.
2. **Optimized:** compile the block, collect guard/recompile diagnostics, and compare static, compiler-dynamic, and explicit bucketed wrappers. Persist a table mapping input signature to cache outcome and latency.
3. **Integrated:** define `torch.export` dynamic dimensions for a deployable vision/audio/text block, validate legal and illegal boundary inputs, then use the same bounds to configure admission/batching in a miniature server.

## Three focused sessions

1. **Observability:** provoke dtype, stride, batch, and sequence changes; read generated guard/recompile diagnostics rather than infer causes from timing.
2. **Policy design:** build a workload histogram, select buckets/ranges, quantify padding work and cold p99, and compare to dynamic compilation.
3. **Constraint testing:** export with declared bounds and make property tests for min, max, just-outside, and relation-violating shapes.

## Experiments and deliverables

Commit `shape_contract.md`, `shape_matrix.csv`, `bucket_policy.yaml`, `test_shape_boundaries.py`, and diagnostics. Record all signatures—not merely sizes—plus dtype, device, strides, grad mode, compiler mode, cache directory, compilation count, CPU compile time, CUDA memory, warm latency, and fallback reason.

## Common failure modes

- Calling only batch/sequence length “shape” while ignoring strides, dtype, and grad state.
- Padding inputs but forgetting masks, positions, output slicing, or KV allocation effects.
- Setting dynamic mode without checking recompiles and calling it solved.
- Benchmarking a warm popular bucket while production sees a long-tail distribution.
- Encoding only current shapes in export constraints and discovering failure after a model upgrade.

## Mastery gate

Given a request-shape histogram and a compiled attention block, choose and justify a dynamic/bucket policy. Demonstrate legal/illegal export bounds, show guard evidence, and quantify the p99 startup/latency versus padding/memory tradeoff with raw measurements.

## Primary sources, code paths, and OSS bridge

- [Dynamic shapes programming model](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html)
- [`torch.compile` dynamic option](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [`torch.export` documentation](https://docs.pytorch.org/docs/stable/export.html)
- [Torch-TensorRT dynamic-shape guide](https://docs.pytorch.org/TensorRT/user_guide/compilation/dynamic_shapes.html)
- [PyTorch source](https://github.com/pytorch/pytorch)

Read `torch/fx/experimental/symbolic_shapes.py`, `torch/_dynamo/guards.py`, `torch/_dynamo/variables/`, `torch/export/`, and relevant tests for your pinned version. Good contributions are minimal guard/recompile regressions, clearer constraint errors, valid dynamic-shape test coverage, or benchmarks that expose a real long-tail workload.

## Next

Module 32 follows captured graphs into functionalization and AOTAutograd, where mutation and backward semantics become compiler input.
