# Project 03 — Optimization Workbench and Custom Operator

**Window:** Weeks 12–25 · **Prerequisites:** Modules 18, 22–42

Choose one hot, well-defined operation from the Transformer project—a linear
epilogue, normalization, RoPE application, small attention subroutine, or
quantization primitive—and build a stack of increasingly integrated versions.

## Required ladder

1. Eager reference with a test oracle and representative shape distribution.
2. PyTorch composition baseline and profiler/roofline hypothesis.
3. FX/`torch.compile` inspection: graph breaks, guards, generated code/trace,
   and a statement of what bottleneck remains.
4. Triton or CUDA implementation with numeric parity, adversarial shapes, and
   performance evidence.
5. A compiler-aware custom-op boundary: schema, fake/meta behavior,
   `torch.library.opcheck`, autograd policy, and export/compile behavior.
6. Native-build dossier: toolchain, architecture flags, cache/build location,
   ABI, logs, peak host RAM, and a clean-environment rerun.

## Required comparisons

Report latency distribution, throughput, peak memory, compile/build time,
correctness tolerance, graph coverage, and unsupported/fallback cases. At least
one test must intentionally exercise a non-contiguous input, dynamic-size case,
or dtype edge that breaks a naive implementation.

## Mastery gate

Explain why your optimized form is legal under the operator contract and why it
is faster for the measured shapes. If it is not faster, state the actual
bottleneck and preserve the negative result. A correct non-speedup is stronger
evidence than a misleading cherry-picked speedup.

**Inference bridge:** Inference 35–44. Module 40's resource-bounded SM120 build
exercise is a mandatory part of the dossier.
