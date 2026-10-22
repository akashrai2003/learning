# Module 37 — Dispatcher, ATen, and Autograd in C++

**Phase:** 6 — Native PyTorch and GPU integration  
**Time:** 12 focused hours  
**Prerequisites:** Module 36, C++ templates/RAII, autograd/aliasing concepts, and registered custom ops.

## Existing inference curriculum bridge

Reinforce [06 — Transformer forward pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), [28 — Quantized kernel integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [41 — PyTorch custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), and [46 — vLLM internals/contributions](../../inference-engineering-curriculum/modules/46-vllm-internals-and-contributions.md).

## Why this matters and outcomes

PyTorch is not a giant `if device == cuda` statement. The dispatcher resolves an operator call through tensor dispatch keys, thread-local modes, and a registered table. Understanding it lets you explain autocast, autograd, FakeTensor, custom backends, quantized tensors, and why a native extension can recurse or bypass gradients accidentally.

You will be able to inspect a dispatch table, reason about key priority and redispatch, register C++ kernels at the correct layer, distinguish ATen’s public operator surface from a backend implementation, and make a small dispatcher-safe contribution.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P37 — Dispatcher, ATen, and autograd in C++**.

1. [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — learn the API or system contract.
2. [C++ API docs](https://docs.pytorch.org/cppdocs/) — study the underlying design, paper, or specification.
3. [ATen native source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch custom-ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch extension-cpp examples](https://github.com/pytorch/extension-cpp) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Each tensor carries a `DispatchKeySet` encoding relevant behavior/backend information. The dispatcher combines tensor keys with thread-local included keys and removes thread-local excluded keys; conceptually:

\[
K_{active}=(K_{tensor}\cup K_{TLS,included})\setminus K_{TLS,excluded}.
\]

It chooses the highest-priority applicable registration for an operator. A CUDA tensor may involve Autograd, Autocast, Functionalize, Python/fake modes, and CUDA behavior—not just `CUDA`. Autocast is typically a thread-local policy key; autograd wraps backend work to build gradient history. A handler that needs to call the next lower implementation must redispatch while excluding its own key, or it can recurse into itself.

ATen is the C++ tensor/operator library beneath Python `torch` APIs. Operator schemas and generated declarations define calls such as `at::add`; native kernels may live in `aten/src/ATen/native`, backend-specific files, or structured kernel implementations. A kernel may be composite (implemented using other ATen ops), CPU/CUDA-specific, or supplied by another dispatch key. Dtype/layout/memory format can affect eligibility and kernels, but they are not all simple substitutes for backend dispatch keys; inspect the actual table rather than reasoning by a guessed switch statement.

For an extension, `TORCH_LIBRARY` defines the namespace/schema and `TORCH_LIBRARY_IMPL(namespace, CUDA, m)` attaches a CUDA implementation. Multiple translation units need appropriate fragment/registration organization; duplicate registrations or static initialization assumptions can cause import-order bugs. Use dispatcher diagnostics in a pinned debug build, but avoid making private `torch._C` inspection calls part of public runtime logic.

Autograd integration has its own contract. A pure composite op can often inherit gradients through constituent ops. A native kernel that bypasses them needs an autograd registration/formula at the appropriate layer, with saved context and correct higher-order behavior assumptions. Calling low-level backend code directly merely to “avoid dispatcher overhead” risks bypassing autocast, functionalization, fake, autograd, device guards, or future compiler transformations; measure before considering such a change.

## Implementation ladder in `torch-lab/`

1. **Reference:** write `torch-lab/dispatcher/inspect.py` to invoke a known op under normal/autocast/grad/FakeTensor contexts and capture supported diagnostic output plus observable dtype/device/grad behavior.
2. **Optimized:** register a tiny `torch_lab::scaled_add` with CPU and CUDA C++ implementations, an optional composite/autograd route, and fake metadata. Dump/annotate its intended registrations and test dispatch on supported modes.
3. **Integrated:** connect the native RMSNorm from Module 36 and show how compiled/autocast/autograd behavior routes through its registrations. Add a capability fallback that calls public ATen/PyTorch operations rather than reaching around the dispatcher.

## Three focused sessions

1. **Key walkthrough:** draw active keys for CPU/CUDA tensors inside/outside autocast and grad contexts. Verify selected behavior with small experiments and source reading.
2. **Registration:** create schema plus CPU/CUDA registrations in separate files; test import ordering, duplicate namespace fragments, and wrong-key failures.
3. **Redispatch/autograd:** inspect a wrapper key conceptually, reproduce a controlled recursion/error in a sandbox, then implement the correct higher-level composition or registration path.

## Experiments and deliverables

Add `dispatch_notes.md`, `dispatch_matrix.csv`, `test_dispatch_modes.py`, `test_autograd_registration.py`, and source diagrams. Document tensor keys/modes, schema, registrations, selected result, dtype/device, aliasing, gradient behavior, and every private diagnostic used solely for learning.

## Common failure modes

- Assuming a CUDA registration automatically covers autograd, autocast, fake, and functionalization.
- Calling the same operator from its own handler without redispatching correctly.
- Registering the same schema/implementation twice due to static initialization/import order.
- Treating `at::` as a stable shortcut around framework semantics.
- Debugging dispatch based on Python names rather than overload/schema identity.

## Mastery gate

Given a custom op that works in eager CUDA but fails under autocast or compile, identify the missing/incorrect registration through dispatch evidence, repair it, and show behavior under CPU, CUDA, fake, and gradient contexts. Explain why bypassing dispatcher would be a regression even if it saves a small measured overhead.

## Primary sources, code paths, and OSS bridge

- [PyTorch dispatcher tutorial](https://pytorch.org/tutorials/advanced/dispatcher.html)
- [Dispatcher DevLog](https://docs.pytorch.org/devlogs/dispatcher/2026-04-16-how-does-the-dispatcher-work/)
- [Extending PyTorch note](https://docs.pytorch.org/docs/stable/notes/extending.html)
- [C++ API documentation](https://docs.pytorch.org/cppdocs/)
- [PyTorch source](https://github.com/pytorch/pytorch)

Read `c10/core/DispatchKey*`, `c10/core/Dispatcher.*`, `aten/src/ATen/core/dispatch/`, `torch/library.h`, and dispatcher tests. Valuable contributions include a missing mode test, a minimal dispatch/registration regression, improved diagnostic text, or a documented schema contract. Do not propose a core dispatcher change without a minimized correctness case and broad test evidence.

## Next

Module 38 traces a core ATen operator from declarative schema through generated glue and backend implementation.
