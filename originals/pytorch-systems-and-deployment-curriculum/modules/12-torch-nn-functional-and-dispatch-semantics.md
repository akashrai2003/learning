# Module 12 — torch.nn.functional and Dispatch Semantics

> **Phase:** I — Engineering foundations  
> **Time:** 9–11 focused hours across three sessions  
> **Prerequisites:** Modules 08–11 and comfort reading a Python call stack

## Existing inference curriculum bridge

Use this module to make the boundary between a layer’s mathematics and its runtime real. It prepares [Module 39 — Triton Kernels](../../inference-engineering-curriculum/modules/39-triton-kernels.md), [Module 41 — PyTorch Custom Operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), and [Module 42 — Dynamo, Inductor, and Triton Compiler Stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md). Those modules optimize and register the operators whose ordinary PyTorch semantics you learn here.

## Why this matters and outcomes

The same operation can appear as a stateful module, a stateless functional call, a public torch API, or a dispatched ATen operator. Confusing these layers leads to missing parameters, incorrect train/eval behavior, unsafe mutation, or a custom operator that works eagerly but fails under autograd or compilation.

By the end, you can choose a Module versus functional API based on state ownership; reproduce a module with a functional call and explicit weights; explain how an operator schema, dispatch key set, and kernel implementation relate; identify mutation/aliasing as first-class semantics; and use a lightweight dispatch trace to guide source reading rather than rely on a guessed call stack.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P12 — torch.nn.functional and dispatch semantics**.

1. [torch.nn.functional](https://docs.pytorch.org/docs/stable/nn.functional.html) — learn the API or system contract.
2. [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — study the underlying design, paper, or specification.
3. [torch.library](https://docs.pytorch.org/docs/stable/library.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ATen native source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

An nn.Module owns state and behavior. torch.nn.functional supplies mostly stateless tensor functions: F.linear takes input, weight, and optional bias explicitly; nn.Linear owns those parameters and calls the underlying functional/operator path. This distinction is valuable for reference implementations, transforms, parameter sharing, and experiments. It is dangerous when state is implicit. For example, F.dropout requires a training argument if you need behavior to follow module mode; nn.Dropout carries that relationship through its Module state.

An operation ultimately has a schema: argument and result types plus semantic facts such as mutation and aliasing. The dispatcher chooses an implementation based on input tensor properties and active modes. Conceptually, a call may pass through handling for autograd, autocast, functionalization, batching, tracing, and then a backend such as CPU or CUDA. The exact stack changes by PyTorch release and operator, but the essential contract does not: a kernel must obey schema, dtype/device/layout constraints, mutation truth, and the rules of every requested transformation.

ATen is the core operator library behind much of torch. A public Python call can decompose into multiple ATen operations, while an aten operator may have device-specific native implementations or a composite implementation. Do not assume one Python function equals one CUDA kernel. Use profiler and source evidence to determine actual work.

Mutation and aliasing are semantic, not merely performance choices. By convention an underscore suffix often denotes in-place mutation, but inspect documentation/schema before drawing a conclusion. An out= argument can reuse caller-provided storage and has autograd restrictions. A view aliases storage without copying. These facts affect autograd version counters, functionalization, graph capture, memory planning, and custom-operator registration. A fast kernel with a false schema is incorrect even if its numerical output looks right.

PyTorch provides two relevant interception layers. __torch_function__ is a Python-level protocol used by tensor-like types at public API boundaries. __torch_dispatch__ operates at the dispatcher level and sees lower-level operations; it powers more advanced tensor subclasses and modes. Neither is a substitute for understanding an operator contract, and both can change performance, tracing, and compiler behavior. Start with observation, then read source before modifying behavior.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/foundations/functional_parity.py. Reproduce nn.Linear, LayerNorm, embedding, and dropout using F APIs and explicitly supplied state. Test forward parity, gradient parity, state ownership, and train/eval behavior.
2. Build a stateful-versus-functional table: operator inputs, owned parameters, buffers, training flag, output aliases, and expected gradients. Include a dropout test that intentionally omits the training argument.

**Optimized**

3. Write a small TorchDispatchMode that logs operator names, shapes, devices, dtypes, and mutation-like calls for one toy forward. Keep it diagnostic only; do not alter results.
4. Make an aliasing/mutation test matrix using add, add_, view, clone, and an out= form where legal. Verify storage identity, autograd response, and numerical parity.

**Integrated**

5. Pick one logged aten operation, trace its Python entry point, native declaration, schema, and backend implementation. Record uncertainty rather than inventing a path.

## Three focused sessions

1. **Stateful versus functional (3h):** implement parity for four layer types, including dropout under both modes. Review which state belongs in a Module and which becomes an explicit argument.
2. **Operator contracts (3h):** construct mutation/alias tests and inspect error behavior with requires_grad inputs. Make a contract table for each operation before checking its implementation.
3. **Dispatch source walk (3h):** collect a small dispatch trace, then follow one operation through native declarations and backend code. Write the path as evidence, not a generic diagram.

## Experiments and deliverables

Commit functional_parity.py, tests, the stateful/functional table, an aliasing matrix, and one dispatch source-walk note. The note must include PyTorch version, exact input properties, observed operator names, source links, and what remains version-dependent. Keep no private dispatcher API as a permanent dependency in your model code.

## Common failure modes

- Replacing a Module with F calls and losing parameter or buffer registration.
- Calling F.dropout without passing the intended training policy.
- Equating a Python API call with one kernel launch.
- Treating underscore naming as a complete mutation/alias specification.
- Using out= or in-place operations in gradient code without checking autograd semantics.
- Using a dispatch mode as a production optimization before testing compiler, autograd, and recursion behavior.

## Mastery gate

Given a layer, state what is owned state, what can be passed functionally, and what behavior changes in train/eval. Given a logged operator, identify its schema, likely dispatch responsibilities, and alias/mutation contract. Demonstrate a parity-tested functional implementation and a correctly scoped diagnostic trace.

## Primary sources and canonical code paths

- [PyTorch torch.nn.functional API](https://docs.pytorch.org/docs/stable/nn.functional.html)
- [PyTorch extending notes](https://docs.pytorch.org/docs/stable/notes/extending.html)
- [PyTorch torch.library API](https://docs.pytorch.org/docs/stable/library.html)
- [PyTorch custom operators tutorial](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- Inspect [functional.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py), [native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml), [DispatchKey.h](https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h), and ATen native kernels.

## OSS bridge and next

A high-quality upstream report here includes a minimal input, device/dtype/layout, expected schema behavior, actual behavior, and a targeted test. Do not diagnose a dispatcher issue merely from a Python stack trace. Module 13 makes the next major state boundary explicit: data ownership, batching, worker processes, and host-to-device handoff.
