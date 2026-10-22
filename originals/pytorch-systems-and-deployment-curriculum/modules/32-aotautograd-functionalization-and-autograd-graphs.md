# Module 32 — AOTAutograd, Functionalization, and Autograd Graphs

**Phase:** 5 — Capture, export, and compiler foundations  
**Time:** 12 focused hours  
**Prerequisites:** autograd mechanics, Modules 29–31, mutation/aliasing basics, and gradient checking.

## Existing inference curriculum bridge

Connect [08 — Residuals and stability](../../inference-engineering-curriculum/modules/08-normalization-residuals-and-stability.md), [39 — Triton kernels](../../inference-engineering-curriculum/modules/39-triton-kernels.md), [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), and [44 — Correctness/profiling](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md).

## Why this matters and outcomes

Inference engineers eventually touch training, QAT, adapters, and custom backward paths. AOTAutograd explains why a program that works eagerly can fail under `torch.compile`: compilers need a functional, traceable account of forward, backward, mutation, views, and saved values. Functionalization is not academic cleanup; it makes state changes explicit enough to transform safely.

You will be able to draw an autograd graph, use `torch.func` to reason about pure parameter functions, identify alias/mutation hazards, validate gradients with finite differences/`gradcheck`, and read AOTAutograd artifacts without depending on unstable private APIs in product code.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P32 — AOTAutograd, functionalization, and autograd graphs**.

1. [AOTAutograd source](https://github.com/pytorch/pytorch/tree/main/torch/_functorch) — learn the API or system contract.
2. [Functionalization source](https://github.com/pytorch/pytorch/tree/main/torch/_functorch) — study the underlying design, paper, or specification.
3. [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch 2 paper](https://arxiv.org/abs/2302.06589) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch compiler docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch Inductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

For \(y=f(x;\theta)\) and scalar loss \(L\), reverse-mode autograd applies vector-Jacobian products from output to leaves:

\[
\frac{\partial L}{\partial \theta}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial \theta}.
\]

Eager PyTorch records a dynamic backward graph and saves tensors needed by backward. An AOTAutograd flow captures a joint forward/backward representation, then partitions it into a forward artifact that returns both user outputs and saved intermediates, and a backward artifact that consumes output gradients and saved values. The partition tradeoff is memory versus recomputation: save a tensor now, or recompute it later. This is related to activation checkpointing but should not be conflated with arbitrary checkpoint APIs.

Mutation and aliasing complicate the transformation. `x.add_(1)` changes storage observed by all aliases; `y=x.view(...)` shares storage and has view metadata. Functionalization rewrites eligible mutation/view programs into equivalent functional operations plus bookkeeping so downstream transforms see explicit inputs/outputs. This requires truthful operator schemas. A custom op that mutates an input while claiming `mutates_args=()` can produce silent wrong answers after reordering or reuse; a false view/alias claim blocks transformations or corrupts behavior.

`torch.func.functional_call` makes parameters/buffers explicit inputs to a module, while transforms such as `grad`, `vmap`, and `jvp` expose mathematical contracts. Use them in labs before relying on compiler internals. `torch.autograd.gradcheck` compares analytical derivatives with numerical finite differences, usually in double precision and carefully selected tolerances; it is a correctness test, not a performance benchmark. Inference-only custom ops still need accurate mutation/alias and FakeTensor contracts if they cross compilation/export boundaries.

## Implementation ladder in `torch-lab/`

1. **Reference:** implement `torch-lab/autograd/reference_graphs.py`: a small residual MLP with explicit parameter dictionary, `torch.func.functional_call`, `torch.func.grad`, and a hand-derived gradient for one linear-plus-MSE example.
2. **Optimized:** add an in-place residual variant and an out-of-place equivalent. Use `gradcheck`, alias/state observations, and a compiled training step to show why a semantic replacement must be proven rather than assumed.
3. **Integrated:** compile a LoRA/QAT-style tiny training step with a registered custom op from Module 35 or 36. Run forward, backward, optimizer update, checkpoint/resume, and eager/compiled gradient parity over contiguous and noncontiguous inputs.

## Three focused sessions

1. **Math to graph:** derive one VJP by hand, inspect `grad_fn`/saved behavior conceptually, and compare it to `torch.func.grad` output on fixed data.
2. **Mutation laboratory:** create views, in-place operations, and a wrong custom-op mutation declaration in a controlled test. Observe eager behavior versus transform/compiler checks; repair the contract.
3. **AOT evidence:** compile a training step, collect graph/trace artifacts, identify forward saved values and backward inputs, and compare activation memory/recompute policy only after gradient parity passes.

## Experiments and deliverables

Add `autograd_contract.md`, `test_gradients.py`, `test_aliasing.py`, `compiled_train_parity.py`, and a graph artifact directory. Record loss function, parameter/input shapes, leaf/nonleaf status, mutation/alias contract, dtype, gradcheck parameters, max forward/gradient/update error, saved activation bytes where observable, and compile configuration.

## Common failure modes

- Using `.data` or unsafe in-place mutation to silence autograd errors.
- Applying `gradcheck` in FP16/BF16 and interpreting numerical noise as a model gradient proof.
- Equating successful eager forward with valid FakeTensor/functionalization behavior.
- Saving every activation without understanding memory, or recomputing everything without measuring runtime.
- Giving a custom op a backward formula that captures data pointers or untracked Python state.

## Mastery gate

For a module containing a view, an in-place update, and a trainable adapter, write the true alias/mutation contract and validate eager/functional/compiled forward and gradients. Explain the forward/backward partition’s saved-versus-recomputed tradeoff with measured memory and time, and reject a tempting but semantically invalid rewrite.

## Primary sources, code paths, and OSS bridge

- [`torch.func` documentation](https://docs.pytorch.org/docs/stable/func.html)
- [Autograd mechanics note](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [gradcheck mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)
- [PyTorch custom operator guidance](https://docs.pytorch.org/docs/stable/library.html)
- [AOTAutograd source](https://github.com/pytorch/pytorch/tree/main/torch/_functorch)

Read `torch/_functorch/aot_autograd.py`, `torch/_functorch/partitioners.py`, `torch/_functorch/functionalization.py`, `torch/autograd/`, and functionalization/AOT tests in the pinned tree. Good contributions are a minimized aliasing/gradient repro, fake/backward test, or a correctness-preserving decomposition with tests—not a speed change that bypasses autograd requirements.

## Next

Module 33 turns captured, functional graphs into fused and scheduled kernels with TorchInductor.
