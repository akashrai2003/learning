# Module 17 — torch.func, vmap, and Functional Transforms

> **Phase:** II — Reliable PyTorch systems  
> **Time:** 9–11 focused hours across three sessions  
> **Prerequisites:** Modules 10–16, reverse/forward-mode AD, and module state contracts

## Existing inference curriculum bridge

Use functional transforms to deepen [Module 03 — Tensor Math, Shapes, and einsum](../../inference-engineering-curriculum/modules/03-tensor-math-shapes-and-einsum.md), [Module 19 — Calibration, Error Metrics, and Evaluation](../../inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md), [Module 22 — GPTQ and Second-Order PTQ](../../inference-engineering-curriculum/modules/22-gptq-second-order-ptq.md), and [Module 41 — PyTorch Custom Operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md). Functional code makes mathematical transforms and compiler-visible semantics easier to test.

## Why this matters and outcomes

Many valuable operations are “apply this model function many ways”: per-sample gradients, ensembles, Jacobian-vector products, influence approximations, gradient checks, and batched evaluation of alternative weights. Writing Python loops works as an oracle but often launches too much work and obscures the actual transformation.

By the end, you can turn a stateful Module into an explicit function of parameters, buffers, and inputs; select vmap, grad, vjp, jvp, jacrev, or jacfwd by output/input geometry; compose transforms safely; and diagnose state, randomness, mutation, and tensor-shape limitations.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P17 — torch.func, vmap, and functional transforms**.

1. [torch.func](https://docs.pytorch.org/docs/stable/func.html) — learn the API or system contract.
2. [UX limitations](https://docs.pytorch.org/docs/stable/func.ux_limitations.html) — study the underlying design, paper, or specification.
3. [Per-sample gradients tutorial](https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch examples](https://github.com/pytorch/examples) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch recipes](https://docs.pytorch.org/tutorials/recipes/recipes_index.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Let f: R^n → R^m have Jacobian J with shape [m,n]. A Jacobian-vector product computes J u for tangent u in R^n; it is natural when input dimension/directions are small or forward sensitivity is required. A vector–Jacobian product computes vᵀJ, conventionally represented with output-shaped cotangent v and input-shaped result Jᵀv; it is reverse mode and underlies ordinary gradient computation. Constructing full J costs memory/work proportional to m*n, so prefer a directional product unless the full matrix is the artifact.

torch.func.functional_call(module, parameter_and_buffer_dicts, args) evaluates a Module with supplied state rather than its resident state. It does not magically make arbitrary code pure. A module that mutates a buffer, depends on global state, performs I/O, or consumes uncontrolled randomness has semantics you must make explicit. Functional transforms work best when f(params, buffers, x) has an explicit output and no hidden changes. Preserve tied-weight policy and buffers; a parameter-only dictionary can be insufficient for BatchNorm-like state.

vmap transforms a function written for one example into a function that maps over declared input dimensions. It is not merely a Python for-loop replacement: batching rules let many operators operate efficiently, but unsupported operations, data-dependent shapes, or side effects can prevent the desired behavior. Specify randomness behavior where applicable and test it. A vmap result’s axis order is part of the contract.

A per-example loss l(theta, x_i, y_i) gives per-sample gradients with vmap(grad(l), in_dims=(None,0,0)), whose parameter leaves carry a leading batch axis. This is useful for clipping or diagnostics, but B copies of gradient leaves can be huge. Compose transforms deliberately: grad(vmap(...)) and vmap(grad(...)) are different mathematical objects. jacrev is typically attractive when m is small relative to n; jacfwd when n is small relative to m. Measure and check numerical results rather than choosing by slogan.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/transforms/functional_mlp.py. Write a tiny pure-ish MLP loss and a manual loop oracle for gradients, per-example losses, and one JVP/VJP.
2. Use functional_call with explicit named parameters and buffers. Test output parity with the original Module and test whether any intentional mutation is surfaced.

**Optimized**

3. Replace the per-example loop with vmap(grad(single_example_loss)). Compare pytree shapes, values, peak memory, and runtime against the oracle.
4. Implement JVP, VJP, jacrev, and jacfwd for small functions. Verify directional products by finite differences and explain when full Jacobians are unnecessary.

**Integrated**

5. Add per-sample gradient-norm diagnostics to the proposed training engine without retaining all gradients in a production-sized batch. Add transform-composition tests under different batch dimensions and randomness policies.
6. Create a source walk of one tensor subclass/custom operator through transformed execution and record unsupported/mutation behavior explicitly.

## Three focused sessions

1. **Functions from modules (3h):** expose parameters/buffers, compare functional_call parity, and identify every hidden state edge.
2. **Differential transforms (3h):** derive JVP/VJP dimensions, implement both plus finite-difference checks, then compare jacrev/jacfwd on controlled shapes.
3. **Batching transforms (3h):** make a loop oracle, apply vmap to per-sample gradients, inspect pytrees/memory, and test randomness/mutation constraints.

## Experiments and deliverables

Commit functional_mlp.py, JVP/VJP tests, vmap parity tests, a transform-composition table, and a per-sample-gradient resource report. The report must give input/output/pytree shapes, transform order, numerical tolerance, peak memory, and why the selected transform is mathematically appropriate.

## Common failure modes

- Forgetting buffers or tied parameters when using functional_call.
- Assuming a stateful/in-place Module is automatically a pure function.
- Building a full Jacobian when a directional product suffices.
- Confusing grad(vmap(f)) with vmap(grad(f)).
- Vectorizing random operations without defining same/different/error behavior.
- Materializing per-sample gradients at a batch size that cannot fit.

## Mastery gate

Given f with stated input/output dimensions, choose JVP, VJP, jacfwd, or jacrev and justify the choice. Convert a Module to a tested functional call with parameters and buffers, then compute per-sample gradients that match a loop oracle and have an explicit memory bound.

## Primary sources and canonical code paths

- [PyTorch torch.func API](https://docs.pytorch.org/docs/stable/func.api)
- [PyTorch functional_call](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html)
- [PyTorch vmap](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html)
- [PyTorch gradcheck mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)
- Inspect [torch/_functorch](https://github.com/pytorch/pytorch/tree/main/torch/_functorch), [torch/func](https://github.com/pytorch/pytorch/tree/main/torch/func), and transform tests.

## OSS bridge and next

Useful contributions include a transform compatibility test, an accurate mutation/error message, or an operator batching rule with a reference oracle. Start from a tiny function and state the intended mathematical transform. Module 18 supplies the observability stack needed to prove where those transforms spend time and memory.
