# Module 42 — Compiler Stack: Dynamo, Inductor, and Triton

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 21, paired with Module 41; three 3-hour sessions plus a generated-code dossier  
**Prerequisites:** Modules 39 and 41; Triton kernels, PyTorch dispatcher/FakeTensor, dynamic shapes, and CUDA graph basics

## Why this matters

`torch.compile took 5 s` compresses several systems into one log line. Dynamo captures Python frames into guarded FX graphs; FakeTensor propagates metadata; functionalization/AOT machinery normalizes operations; Inductor schedules, fuses, and lowers; Triton or C++ generates kernels; native toolchains build artifacts; caches and CUDA graphs may add later warm-up. Knowing the boundary turns “compile crashed” into a testable stage and prevents eager success from being mistaken for compiled correctness.

## Outcomes

You can:

- trace a function from Python bytecode to FX and generated GPU code;
- explain graph breaks, guards, specialization, dynamic shapes, and recompilation;
- make a custom operator visible to compilation through accurate FakeTensor metadata;
- inspect Inductor/Triton caches and separate compile, autotune, graph capture, and execution;
- diagnose whether a failure belongs to capture, lowering, code generation, build, load, or launch.

## Mental model: specialize, lower, cache, guard

Dynamo hooks CPython frame evaluation and symbolically executes bytecode. Capturable tensor work becomes an FX graph; unsupported or data-dependent Python can cause a graph break. The compiled callable is valid only while guards hold. Typical guards cover Python identities/constants and tensor device, dtype, rank, sizes, and strides.

If a guard fails, Dynamo chooses another cached graph or recompiles. With lengths \(L_1,L_2,\ldots\), fully static specialization may produce one artifact per shape. Symbolic dimensions can reduce variants but require every operation and backend to support them. “Dynamic” is not “no guards”; ranges, relations, strides, and control flow still matter.

The simplified inference path is:

```text
Python frame -> Dynamo/FX -> FakeTensor shape propagation
 -> functional graph and decompositions -> Inductor scheduling/fusion
 -> Triton/C++ source -> compiler/linker -> cached module -> guarded execution
```

Custom operators may be opaque graph nodes when their fake implementation describes outputs. Decompositions instead expose internal operations to fusion, but also enlarge compiler responsibility. Choose opacity when the operator already owns an optimized kernel or has semantics the compiler should not rewrite; choose a decomposition when composition/fusion is valuable and stable.

Inductor groups compatible pointwise/reduction work, plans memory, chooses templates or generated kernels, and commonly emits Triton for GPUs. Generated code has launch grids, block sizes, warps, stages, guards, and sometimes autotuning variants. Inspect it rather than assuming fusion occurred. CUDA graph capture is later execution optimization and can fail after successful compilation while warming shapes or reserving stable memory.

## Connection to the SM120 runtime

In the learner's RTX 5060 Ti experiment, the model graph, Triton attention, and FlashInfer NVFP4 custom operator can coexist. Inductor may generate some kernels while dispatching the NVFP4 node into a JIT-built shared object. The compiler's cache and FlashInfer's JIT cache are distinct. Both need architecture and ABI-correct artifacts; deleting one cache does not invalidate the other.

Likewise, a log ending after successful `torch.compile` does not identify compilation as the crash. The next stage may profile memory, autotune, warm a custom kernel, or capture many CUDA graph sizes. Capture range, process exit status, and system OOM evidence before assigning blame.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/compiler_trace.py` with a small dynamic-shape SwiGLU/normalization function. Export eager outputs, FX graphs, guard/recompile logs, and a table of shapes/strides that share or miss a compiled artifact.

### 2. Optimized

Compile the custom op from Module 41 both as an opaque registered op and as a decomposed PyTorch expression. Inspect generated Triton/C++ source, count kernels and materialized intermediates, and compare compilation latency, memory, and warm execution.

### 3. Integrated

Instrument a decoder layer in `inference-lab/engine/compiled_decoder.py`. Vary prefill length and decode batch, bound shape buckets, and compare eager, compiled, and compiled-plus-CUDA-graph modes. Archive cache keys and generated source for one exact run without assuming those internals are a stable API.

## Three 3-hour sessions

1. **Capture and guards:** trigger graph breaks and recompilations intentionally; use full-graph mode and compiler logs to explain each.
2. **Lowering and codegen:** trace FX through Inductor, inspect generated Triton, and verify whether expected operations fused.
3. **Engine integration:** compile the decoder/custom op across real shape regimes, separate cold/warm phases, and reproduce one failure at a named stage.

## Experiments and deliverables

Commit:

- `reference/compiler_trace.py`;
- `engine/compiled_decoder.py`;
- `tests/test_compiled_custom_op.py`;
- `benchmarks/results/compiler-shape-cache.json`;
- `notes/generated_code/decoder-kernel.md`;
- `notes/failure_postmortems/compile-stage-triage.md`.

Record PyTorch/Triton commits, compile mode/options, environment logs, FX graph hashes, guards, graph breaks, recompiles, generated kernel count, compile/autotune/capture time, cache state, peak host/GPU memory, and warm latency.

## Common failure modes

- Calling every post-Dynamo failure a `torch.compile` error.
- Suppressing compiler errors and benchmarking an eager fallback unknowingly.
- Adding `allow_in_graph` without understanding downstream tracing.
- Returning inaccurate FakeTensor metadata.
- Specializing on every prompt length and causing a compile storm.
- Comparing cold compiled execution with warm eager execution.
- Assuming an FX fusion became one GPU kernel.
- Deleting unrelated caches or all caches before preserving failure evidence.
- Treating cache reuse across toolchain/SM changes as valid.

## Mastery gate

Given a compiled function and five input shapes, predict graph variants and guards, then verify with trace logs. Locate the generated kernel, explain its grid and memory traffic, and show numerical parity. Integrate the Module 41 op in full-graph mode and classify an injected failure as capture, lowering, native build, load, or launch using direct evidence.

## Canonical code areas and OSS bridge

Study `torch/_dynamo/`, `torch/fx/`, `torch/_subclasses/fake_tensor.py`, `torch/_functorch/aot_autograd.py`, and `torch/_inductor/` in the [PyTorch repository](https://github.com/pytorch/pytorch). In vLLM, follow `vllm/compilation/`, compilation configuration, custom-op boundaries, and CUDA graph management. Useful contributions include minimal graph-break reproducers, fake/meta fixes, safe decompositions, dynamic-shape tests, or cache diagnostics with a pinned generated-code artifact.

## Primary and authoritative sources

- [`torch.compile` programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)
- [Dynamo overview](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)
- [`torch.compile` troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
- [TorchInductor source](https://github.com/pytorch/pytorch/tree/main/torch/_inductor)
- [vLLM `torch.compile` integration design](https://docs.vllm.ai/en/latest/design/torch_compile/)

## Next

Module 43 opens the native-build box: emitted Ninja rules, CUDA compiler parallelism, shared objects, dynamic linking, and ABI compatibility.
