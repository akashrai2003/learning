# Module 11 — Modules, Parameters, Buffers, Hooks, and State

> **Phase:** I — Engineering foundations  
> **Time:** 8–10 focused hours across three sessions  
> **Prerequisites:** Modules 08–10 and object-oriented Python

## Existing inference curriculum bridge

Apply this module while rebuilding pieces of [Module 06 — Transformer Forward Pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), [Module 07 — Tokenization, Embeddings, and Logits](../../inference-engineering-curriculum/modules/07-tokenization-embeddings-and-logits.md), and [Module 08 — Normalization, Residuals, and Stability](../../inference-engineering-curriculum/modules/08-normalization-residuals-and-stability.md). Their mathematical layers become portable, trainable, and serializable only when PyTorch state registration is correct.

## Why this matters and outcomes

nn.Module is not merely a class with forward. It is PyTorch’s recursive state and behavior boundary: it discovers submodules, registers trainable parameters and persistent buffers, moves state across devices, switches train/eval behavior, emits a state dictionary, and routes module hooks. A plain Python list or an ordinary Tensor can look correct in one eager run and quietly evade the optimizer, checkpoint, or device move.

By the end, you can design stateful modules with explicit contracts; choose Parameter versus persistent/non-persistent buffer versus ordinary attribute; preserve parameter sharing; use containers that register state; inspect a state dictionary; and use hooks as bounded diagnostic tools without making them hidden model semantics.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P11 — Modules, parameters, buffers, hooks, and state**.

1. [torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) — learn the API or system contract.
2. [Saving/loading models](https://docs.pytorch.org/docs/stable/notes/serialization.html) — study the underlying design, paper, or specification.
3. [Module hooks](https://docs.pytorch.org/docs/stable/notes/modules.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch tensor views note](https://docs.pytorch.org/docs/stable/tensor_view.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ATen native source](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Module assignment is registration. Assigning an nn.Module as an attribute makes it a child module; assigning nn.Parameter makes it a trainable parameter; register_buffer gives a Tensor module state that is not optimized. An ordinary Tensor attribute is just Python state: model.to(device), model.parameters(), and state_dict do not automatically manage it. This distinction is foundational.

A parameter is differentiable model state, usually weight or bias. A buffer is model state with no optimizer update, such as BatchNorm running statistics, a fixed positional table, an attention mask cache, or a quantization scale. Persistent buffers enter state_dict; non-persistent buffers move with device/dtype conversions but are omitted from state_dict, appropriate for derived or rebuildable caches. Decide persistence from the artifact contract, not because a tensor happens to be convenient.

Modules recurse through registered children. Use ModuleList and ModuleDict for variable collections of layers and ParameterList or ParameterDict for raw trainable tensors. A normal list/dict does not perform registration. Tied embeddings and output weights are intentionally the same Parameter object; preserve that identity rather than copying values. Named traversal normally deduplicates repeated module instances, while state_dict preserves qualified state keys. Build tests around actual identities and keys.

Calling module(x) invokes Module’s call path, which coordinates forward, registered hooks, and other framework behavior. Calling module.forward(x) directly bypasses this wrapper and is almost always the wrong application-level choice. Module.train() and eval() recurse through children and update the training flag; they do not change autograd mode. State transitions should be visible in tests.

state_dict is a mapping of parameter and persistent-buffer names to tensor values. Treat it as a defined artifact boundary, but remember that its tensor values are shallow references at the time it is produced; an in-memory “best checkpoint” needs cloned/copied data or immediate serialization. Module 16 will make saving/loading a full, secure, versioned operation.

Hooks are instrumentation or transformation points around forward/backward. Register them with a removable handle and remove them deterministically. A forward hook can observe or even replace output, but hidden behavior in hooks makes reasoning, export, and compilation harder. Do not use hooks as a primary model architecture mechanism. Full backward hooks have extra aliasing and mutation constraints; use them only after reading their exact API contract.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/models/stateful_block.py with a small residual block. Register weights as Parameters, a fixed scale as a persistent buffer, and a derived causal-mask cache as non-persistent.
2. Write tests for parameters(), named_buffers(), state_dict keys, device/dtype moves, and train/eval propagation. Include an intentional ordinary-Tensor attribute bug and show the failed move/checkpoint test.

**Optimized**

3. Make a variable-depth model first with a Python list, then ModuleList. Prove the former has missing parameters and the latter has expected names and optimizer coverage.
4. Add tied input/output embeddings and assert object identity, gradient behavior, and state restoration behavior.

**Integrated**

5. Implement a temporary forward hook that collects activation shape/dtype/finite statistics. Ensure it is removed in a finally block and verify outputs are unchanged.

## Three focused sessions

1. **State registration (3h):** implement the residual block, inspect recursive names, and make parameter/buffer/device tests pass. Explain every state member in a table.
2. **Containers and sharing (3h):** create the unregistered-list failure, fix it with ModuleList, then add tied weights. Compare object identity, state keys, and optimizer parameter groups.
3. **Hooks and artifact boundaries (3h):** implement observability hooks, test exception-safe removal, and make a state-dictionary snapshot experiment. Capture why a shallow reference is not an immutable checkpoint.

## Experiments and deliverables

Commit stateful_block.py, tests, a state inventory table, a tied-weight test, and a hook report. The inventory must list name, type, shape, dtype, device, trainability, persistence, serialization behavior, and reason for existence. Include a test that changes model.to(dtype/device) and verifies all intended state moved.

## Common failure modes

- Storing layers in a Python list/dict instead of ModuleList/ModuleDict.
- Storing an intended constant as an ordinary Tensor instead of a buffer.
- Using a buffer for a quantity that should receive gradients.
- Calling forward directly and silently skipping hooks/call-path behavior.
- Replacing tied weights with equal but distinct Parameters.
- Leaving hooks installed across tests, requests, or compiled execution.
- Keeping an in-memory state_dict reference and mistaking it for a frozen checkpoint.

## Mastery gate

Given a model attribute, classify it as parameter, persistent buffer, non-persistent buffer, child module, or ordinary Python configuration, and defend the choice. Build a model whose device moves, optimizer coverage, state keys, sharing, and hooks are all verified by tests. Explain why each state entry is serialized or rebuilt.

## Primary sources and canonical code paths

- [PyTorch nn.Module API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch Parameter API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)
- [PyTorch hooks API](https://docs.pytorch.org/docs/stable/notes/modules.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- Inspect [module.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py), [parameter.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/parameter.py), and [hooks.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/hooks.py).

## OSS bridge and next

Good PyTorch contributions in this area are missing state/move tests, corrected buffer persistence, or documentation that clarifies hook ordering and scope. First demonstrate the issue with a tiny module. Module 12 moves below Module into stateless functional operators, ATen dispatch, mutation, and transformation semantics.
