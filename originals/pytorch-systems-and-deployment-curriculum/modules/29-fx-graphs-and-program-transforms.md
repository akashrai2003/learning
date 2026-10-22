# Module 29 — FX Graphs and Program Transforms

**Phase:** 5 — Capture, export, and compiler foundations  
**Time:** 12 focused hours  
**Prerequisites:** Python object model, `nn.Module`, tensor shapes/strides, and evaluation discipline.

## Existing inference curriculum bridge

Bridge to [06 — Transformer forward pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), [17 — Architecture forensics](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md), [38 — Tiling, fusion, and pipelining](../../inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md), [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), and [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md).

## Why this matters and outcomes

FX is the graph vocabulary underlying much of modern PyTorch tooling. It gives you a concrete answer to “what does this Python module actually compute?” and lets you write transformations as explicit rewrites rather than untestable monkey patches. Its limits also teach an essential lesson: a graph capture is a program representation with assumptions, not a universal serialization of Python.

You will be able to read and print FX IR, trace a suitable module, safely write a small graph pass, validate semantic and metadata parity, distinguish FX symbolic tracing from Dynamo/export capture, and use graphs as evidence while debugging compiler or deployment behavior.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P29 — FX graphs and program transforms**.

1. [FX documentation](https://docs.pytorch.org/docs/stable/fx.html) — learn the API or system contract.
2. [FX limitations](https://docs.pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing) — study the underlying design, paper, or specification.
3. [FX graph mode quantization](https://docs.pytorch.org/docs/stable/quantization.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

An FX `GraphModule` holds a graph plus a module hierarchy. Nodes generally have operation kinds `placeholder`, `get_attr`, `call_function`, `call_method`, `call_module`, and `output`. A toy block

```python
def f(x, w):
    return torch.relu(x @ w + 1)
```

becomes a dependency DAG with nodes for `matmul`, `add`, and `relu`; node arguments point to prior nodes or constants. Its graph is not the tensor values, and a node’s `meta` is advisory unless your pass validates it. `get_attr` may reference parameters/buffers stored on the generated module; rewriting those carelessly can change state semantics.

`torch.fx.symbolic_trace` executes Python with proxy objects. It works well for many module/dataflow patterns but cannot generally decide arbitrary Python control flow depending on a proxy tensor. Dynamo observes CPython bytecode at runtime, and `torch.export` produces a more constrained ahead-of-time program with explicit shape constraints. Use the representation demanded by the task; never “solve” a capture limitation by specializing a hidden production input without documenting it.

A safe transform has a contract: target patterns, type/shape/layout assumptions, alias/mutation behavior, numerical tolerance, and fallback. For example, recognizing `linear → bias → SiLU` may enable an epilogue fusion only when dimensions/dtypes are supported. The original graph is your oracle. Recompile the `GraphModule`, lint the graph, preserve topological order, use `node.replace_all_uses_with` only after observing users, and test random plus adversarial inputs. Rewrite names and `stack_trace` metadata help diagnostics but do not substitute for tests.

## Implementation ladder in `torch-lab/`

1. **Reference:** make `torch-lab/compiler/fx_inspect.py` trace a small MLP and decoder block, emit readable graph/code, and save a JSON summary of nodes, targets, parameter ownership, and example tensor metadata.
2. **Optimized:** write `torch-lab/compiler/fx_passes.py` containing one conservative pass, such as eliminating `x + 0` only where the zero is an immutable scalar and dtype semantics match. Add `graph.lint()`, `recompile()`, parity tests, and a no-op fallback when the pattern is absent.
3. **Integrated:** inspect the same block through symbolic FX, `torch.export`, and `torch.compile` diagnostics. Relate the captured operations to an Inductor fusion or a custom-op boundary; do not assume identical graphs represent a bug.

## Three focused sessions

1. **IR reading:** trace several modules and annotate each node against source, including `get_attr`, modules, methods, and explicit constants. Intentionally use tensor-dependent control flow and document why symbolic tracing fails.
2. **Pass correctness:** build a pattern/rewrite, test shapes `(0, …)`, noncontiguous inputs, dtype variants, aliases, and gradients where relevant. Add metamorphic tests, not only fixed expected values.
3. **Capture comparison:** export/compile an equivalent block, inspect guards/constraints or graph breaks, and write when FX is the right tool versus a wrong one.

## Experiments and deliverables

Commit `graphs/*.py`, `graphs/*.svg` or text dumps, `fx_passes.py`, `test_fx_passes.py`, and `capture_comparison.md`. Record PyTorch revision, trace inputs, source hash, graph hash, node count, metadata source, transformation preconditions, numerical/gradient tolerances, and unsupported cases.

## Common failure modes

- Treating a traced graph as a proof that all Python branches were captured.
- Rewriting an operation that has mutation, alias, randomness, dtype-promotion, or numerical-order implications.
- Trusting shape metadata propagated from a different device/dtype than runtime.
- Calling `GraphModule.recompile()` optional after editing a graph.
- Using textual node names as stable public APIs across releases.

## Mastery gate

Given a module and a proposed fusion, produce the original graph, a minimal legal rewrite, a precondition checker, and forward/gradient parity tests. Explain a case where symbolic tracing is invalid but export or Dynamo capture is useful, and a case where no graph transformation should be attempted.

## Primary sources, code paths, and OSS bridge

- [FX documentation](https://docs.pytorch.org/docs/stable/fx.html)
- [FX API reference](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.Graph)
- [`torch.export` documentation](https://docs.pytorch.org/docs/stable/export.html)
- [PyTorch custom operators guidance](https://docs.pytorch.org/docs/stable/library.html)
- [PyTorch source](https://github.com/pytorch/pytorch)

Read `torch/fx/`, `torch/fx/passes/`, `torch/_functorch/` capture utilities, and tests in the pinned checkout. Good OSS entry points are a minimal tracer/capture reproducer, a regression test for a conservative pass, better error context, or a correctness test for an existing transform. Avoid changing a general rewrite based only on one model trace.

## Next

Module 30 explains how `torch.compile` obtains graph regions from ordinary Python execution.
