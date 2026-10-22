# Module 28 — Quantized Kernel and System Integration

**Phase:** 4 — Quantization  
**Pacing:** Week 14, paired with Module 27; three 3-hour sessions plus the Phase 4 gate  
**Prerequisites:** Modules 17–27; model forensics, quantized formats, evaluation, GEMM/attention shapes, and reproducible environments

## Why this matters

Most quantization failures happen between the algorithm and the multiply: metadata names do not match layouts, loaders repack incorrectly, a kernel excludes one shape or architecture, compilation uses the wrong target, or an engine silently falls back. Integration means preserving one numerical and physical contract from checkpoint bytes through parameter transforms, dispatch, compiled graph, and serving metrics.

## Outcomes

You can:

- specify a complete quantized operator ABI;
- trace serialization → loader → repacker → dispatcher → kernel;
- implement correctness and capability gates before performance tuning;
- diagnose architecture, JIT, linker, and mixed-format failures;
- demonstrate end-to-end quality and speed without hiding fallbacks.

## Mental model: five contracts must agree

For a quantized linear,

\[
Y_{[M,N]}=
\operatorname{epilogue}\left(
X_{[M,K]},
\operatorname{unpack}(Q_{[N,K]},S,Z)^\top
\right).
\]

The implementation is correct only when these contracts agree:

1. **Numerical:** codebook/integer range, \(s,z\), grouping, rounding, clipping, accumulator and output dtype.
2. **Physical:** packed shape, byte order, scale layout, permutation, padding, alignment.
3. **Operator:** \(M,N,K\), transposition, bias/fused activation, tensor-parallel shard axis, MoE grouping.
4. **Hardware:** compute capability, instructions, toolkit/compiler target, shared-memory and alignment limits.
5. **Runtime:** checkpoint schema, model mapping, kernel registry, graph capture/compile, fallback and error policy.

A useful format descriptor is a versioned tuple such as:

```text
(method, bits, element_format, group_axis, group_size,
 scale_dtype, zero_point_domain, pack_layout, pack_version,
 logical_shape, padded_shape, permutation, accumulator_dtype)
```

“W4A16” is not enough.

## Dispatch as a proof obligation

Kernel selection should answer in order:

1. Is the format semantically supported?
2. Is this GPU/toolkit build supported?
3. Can the kernel implement these exact shapes/layouts?
4. Is workspace available?
5. Which candidate wins for the actual \(M\) regime?

Mixed checkpoints may need different kernels for FP8 attention projections, NVFP4 MLPs, unquantized embeddings, and MoE experts. A global forced backend can remove every valid candidate for one layer family. Auto-dispatch still needs logs and tests.

Loading may transform parameters once: fuse QKV, transpose, shard, permute, pad, or prepack. These transformations should be pure, versioned, independently testable, and never repeated per request. Preserve the original logical descriptor for diagnostics.

## Correctness before speed

Test three levels:

- **Bit/format:** golden packed bytes and qparam tensors.
- **Operator:** optimized kernel versus a high-precision explicit dequant oracle across supported and rejected shapes.
- **Model:** layer outputs, logit KL/top-\(k\), PPL/tasks, and cached decode.

Use error tolerances justified by format and accumulation, plus NaN/Inf/saturation checks. Test noncontiguous inputs, tails, odd batches, graph capture, multiple streams, tensor-parallel shards, and deterministic repeated calls.

For performance, separate cold conversion/JIT, warm microkernel, model prefill, model decode, and aggregate serving. A kernel microbenchmark does not prove end-to-end improvement.

## Implementation ladder

### 1. Reference

Create `inference-lab/engine/quantization/contract.py` and a registry with explicit `supports_format`, `supports_device`, `can_implement`, `prepare_weights`, and `apply`. Add a slow oracle backend that accepts every valid logical format.

### 2. Optimized

Integrate one packed low-bit kernel. Cache prepared weights by format/version/device, add shape-keyed selection, and benchmark candidate crossover points. Emit structured dispatch reasons rather than a boolean.

### 3. Integrated

Load a mixed-format model, capture a complete dispatch manifest per layer, compile/capture graphs, and serve reproducible prefill/decode workloads. Inject failures: wrong pack version, unsupported SM, misaligned K, missing scale, corrupt JIT cache, and unavailable shared library. Each must fail safely or select a declared fallback.

## Three 3-hour sessions

1. **Contract and tests:** define descriptor/registry, implement the oracle, and create golden plus negative format tests.
2. **Kernel integration:** add prepare/apply, architecture and shape capability checks, profiler ranges, and cold/warm benchmarks.
3. **Engine source walk:** trace an actual model from quant config through parameter loading, kernel selection, compilation/JIT, CUDA launch, and fallback logging.

The weekend Phase 4 report recommends a format for one model/hardware/workload and defends calibration, quality, packing, kernel, memory, latency, and throughput evidence.

## Experiments and deliverables

Commit:

- `engine/quantization/contract.py`;
- `engine/quantization/registry.py`;
- `tests/test_quantized_dispatch.py`;
- `benchmarks/manifests/quantized-engine.yaml`;
- `benchmarks/results/quantized-kernel-system.json`;
- `notes/failure_postmortems/quantized-integration.md`.

Capture environment manifest, GPU SM, driver/toolkit, framework/runtime/kernel-library commits, compiler flags, binary architecture list, checkpoint hash, dispatch log, warm-up, raw results, and baseline.

## Common failure modes

- Letting method names substitute for a versioned layout contract.
- Correct loader plus wrong kernel transpose/group axis.
- Applying checkpoint scales twice or defaulting missing scales silently.
- Forcing one backend across mixed quantization families.
- Compiling for a related but binary-incompatible GPU architecture.
- Treating a JIT/build crash as model OOM without process/system evidence.
- Benchmarking after fallback without checking selected kernels.
- Passing microkernel tests while PPL or cached decode regresses.
- Mixing cold compile time with steady-state latency inconsistently.

## Mastery gate

Given a quantized checkpoint and target GPU, produce a layer-by-layer format/dispatch manifest, run golden operator and model-quality tests, explain every fallback, and present cold/warm prefill/decode benchmarks. Reproduce and diagnose one intentionally broken integration without changing multiple variables at once.

## Canonical code paths and OSS bridge

In vLLM, walk `model_executor/layers/quantization/` into `model_executor/kernels/linear/`, then attention backend selection, platform capability checks, and compilation config. Compare the [vLLM repository](https://github.com/vllm-project/vllm), [compressed-tensors schema/loader](https://github.com/neuralmagic/compressed-tensors), [CUTLASS](https://github.com/NVIDIA/cutlass), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), and TensorRT-LLM quant plugins. Strong upstream contributions include minimal reproducers, capability diagnostics, golden format tests, missing shape cases, and benchmarks with declared dispatch.

## Primary and authoritative sources

- [vLLM source](https://github.com/vllm-project/vllm)
- [compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [TensorRT-LLM quantization examples](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md)

## Next

Module 29 starts Phase 5 by deriving exact attention as an IO problem and building the online-softmax recurrence that makes tiled attention possible.
