# Module 10 — Autograd Engine and Gradient Modes

> **Phase:** I — Engineering foundations  
> **Time:** 8–10 focused hours across three sessions  
> **Prerequisites:** Modules 08–09 and multivariable chain rule

## Existing inference curriculum bridge

Contrast a fine-tuning path with [Module 06 — Transformer Forward Pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), and connect low-precision gradient behavior to [Module 04 — Numerical Formats and Error](../../inference-engineering-curriculum/modules/04-numerical-formats-and-error.md). Later, [Module 41 — PyTorch Custom Operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md) depends on the ability to state autograd and mutation semantics precisely.

## Why this matters and outcomes

Autograd is the contract that lets PyTorch compose models, transformations, mixed precision, distributed training, and custom operators. It is also why a training run uses far more memory than inference, and why an apparently benign in-place update can fail much later in backward.

By the end, you can trace a dynamic autograd graph from output to leaves; derive and verify vector–Jacobian products; choose between backward and autograd.grad; use train/eval, no_grad, and inference_mode for their distinct purposes; and diagnose in-place, detach, retained-graph, version-counter, and saved-tensor failures.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P10 — Autograd engine and gradient modes**.

1. [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — learn the API or system contract.
2. [Autograd API](https://docs.pytorch.org/docs/stable/autograd.html) — study the underlying design, paper, or specification.
3. [Extending autograd](https://docs.pytorch.org/docs/stable/notes/extending.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ATen native source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

For scalar loss L and parameter w, reverse-mode automatic differentiation computes dL/dw by applying the chain rule backward. In the general case, calling backward with seed v computes a vector–Jacobian product vᵀJ. A non-scalar output therefore needs a seed gradient or an equivalent reduction. Reverse mode is efficient when the output is small and parameters are numerous, which is exactly the common learning setup.

PyTorch executes eager forwards and records a dynamic graph of backward Function nodes only when grad mode and input requirements warrant it. Each node may save input, output, metadata, or other values necessary for its derivative. The graph is normally released after backward to reclaim memory. Retaining it or building a graph through backward with create_graph=True is deliberate higher-order work, not a default cure for an error.

A leaf is usually user-created with requires_grad=True. Intermediate outputs are non-leaves and ordinarily do not retain a .grad field. loss.backward() accumulates into leaf gradient buffers rather than overwriting them, so a training loop must clear gradients before the next accumulation window. zero_grad(set_to_none=True) avoids a zero fill and makes accidentally skipped backward visible, but every gradient consumer must tolerate None.

autograd.grad returns specified derivatives without populating leaf .grad by default. It is suited to probes, penalties, and higher-order derivation. Use it intentionally: it can still retain or construct costly graphs depending on arguments. Compare its result with analytical derivatives and directional finite differences, preferably in float64 on a small test.

Three controls are commonly confused:

- module.train() and module.eval() alter module behavior such as dropout and batch norm; they do not control graph recording.
- torch.no_grad() disables ordinary autograd recording and is the common validation or serving context.
- torch.inference_mode() is a stronger inference-focused mode that also removes selected autograd/view bookkeeping. Tensors created inside it have restrictions, so do not casually carry them into a later gradient-tracked section.

Autograd maintains version counters for tensors saved by backward. An in-place mutation of a required tensor or aliased view can trigger a version-mismatch error. That error is valuable: it prevents a silently wrong derivative. detach makes a tensor sharing storage but disconnected from the current graph; changing either alias still affects data. Using .data circumvents safeguards and is not a valid training technique.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/autograd/trace_graph.py. Run a tiny MLP and record requires_grad, is_leaf, grad_fn, and leaf gradients after backward.
2. Derive gradients for L = mean((Xw-y)^2) by hand. Compare an FP64 implementation with autograd and with directional finite differences.

**Optimized**

3. Demonstrate accumulation, set_to_none, autograd.grad, scalar/non-scalar output, and explicit grad_outputs. Log which .grad buffers change in each case.
4. Create minimal failures for an in-place saved tensor, unintended detach, second backward after release, and validation performed without no_grad. Explain the graph/lifetime cause before repairing each.

**Integrated**

5. Write a mode-matrix test for train/eval crossed with grad-enabled, no_grad, and inference_mode. Use dropout and a small normalization module to show that mode and grad are independent axes.

## Three focused sessions

1. **Reverse mode on paper and in code (3h):** draw a graph, calculate a vector–Jacobian product, compare FP64 autograd with finite differences, then inspect gradient accumulation.
2. **Modes and memory (3h):** execute the mode matrix, measure saved activation memory, and prove that eval alone still records a graph. Inspect tensors created inside inference mode.
3. **Failure forensics (3h):** trigger an in-place/version error and use anomaly detection only to locate it. Make a minimized test and then disable anomaly detection for performance work.

## Experiments and deliverables

Commit graph sketches, a derivative script, finite-difference tests, a mode matrix, and a short failure dossier. For each experiment record which values backward needs, which graph is released, and which gradient storage is mutated. Include one test that would catch an accidental no_grad scope around a trainable subgraph.

## Common failure modes

- Forgetting that backward accumulates gradients rather than resetting them.
- Calling eval and assuming it removes activation storage.
- Retaining a graph to conceal an unintended double backward.
- Mutating a tensor or view saved for backward.
- Using .data or detached aliases to evade a version error.
- Enabling anomaly detection during a throughput benchmark.

## Mastery gate

Given a forward expression, derive its vector–Jacobian product, identify leaves and saved values, and predict all changed .grad fields. Repair a minimal in-place autograd failure without .data or an unnecessary retained graph. Explain when serving should use no_grad versus inference_mode.

## Primary sources and canonical code paths

- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [PyTorch no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
- [PyTorch inference_mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
- [PyTorch autograd.grad](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)
- [PyTorch extending autograd](https://docs.pytorch.org/docs/stable/notes/extending.html)
- Read [engine.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/engine.cpp), [function.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/function.h), and the repository’s autograd tests.

## OSS bridge and next

Useful early work includes a precise autograd regression test, an improved error message, or a minimal issue report with graph and version information. Do not start with an engine rewrite. Module 11 packages tensors and autograd behavior into modules, parameters, buffers, hooks, and recursive state.
