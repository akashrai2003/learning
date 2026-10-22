# Local RTX 5060 Ti Lab Track

The 16 GB RTX 5060 Ti is a strong learning device because it combines current
Blackwell low-precision support with constraints that make memory accounting,
kernel availability, cache policy, and compile behavior impossible to ignore.
It is not a substitute for datacenter multi-GPU hardware, and its compute
capability 12.0 resource limits and supported features must be checked rather
than inferred from B200 examples.

Use the [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
and the installed CUDA toolkit's documentation as the authority for target
details.

## Keep two model classes

### Small semantic models

Use a sub-billion or low-single-digit-billion model for:

- checkpoint mapping;
- full-versus-cached parity;
- quantizing one layer at a time;
- repeated quality tests;
- quick engine restarts;
- custom operator integration.

The purpose is fast falsification and good coverage, not impressive throughput.

### The Gemma 4 NVFP4 systems model

Use the working NVFP4 model for:

- real mixed FP8/NVFP4 dispatch;
- FlashInfer/CUTLASS JIT inspection;
- host and VRAM pressure;
- CUDA graph/warm-up behavior;
- serving throughput versus latency;
- long-context KV-cache experiments;
- production runtime source walks.

Do not use a 9+ GiB resident model as the only way to test a kernel.

## Local exercise ladder

| Modules | Exercise on this machine | Evidence |
|---|---|---|
| 01–02 | Trace the working vLLM request and capture the complete binary/build manifest | process map, shared libraries, cache locations, selected kernels |
| 03–04 | Implement shape and float-format sandboxes on CPU and GPU | bit patterns, rounding, accumulation error |
| 05 | Measure device memory bandwidth and compute roof proxies | roofline estimates versus microbenchmarks |
| 06–17 | Run reference model work on a small checkpoint | layer/logit/cache parity and architecture dossier |
| 18–24 | Quantize small-model layers with INT8/INT4/GPTQ/AWQ/NF4-style formats | error/quality/size reports |
| 25–28 | Emulate FP8/FP4, then inspect the native NVFP4 model path | scales, layouts, metadata overhead, dispatch and fallback |
| 29–34 | Compare reference, PyTorch SDPA, and available attention/runtime paths | prefill/decode latency, cache memory, scheduler traces |
| 35–36 | Query device properties; compile `sm_120`; inspect PTX and SASS | resource table, binaries, microbenchmarks |
| 37–40 | Implement GEMM/normalization in CUDA, Triton, and a CUTLASS example | correctness plus profiler-guided shape sweep |
| 41–43 | Package one custom op and recreate the FlashInfer-style JIT chain | dispatcher, FakeTensor, Ninja file, shared library, ABI report |
| 44 | Benchmark stable versus experimental configurations | raw vLLM and Nsight reports |
| 45 | Simulate communication; run local multi-process collectives where meaningful | analytical model clearly separated from real multi-GPU claims |
| 46–48 | Reproduce a runtime issue or portability gap on `sm_120` | upstream-quality dossier and patch |

## Reuse the successful JIT incident

Preserve a sanitized copy of:

- the generated `build.ninja`;
- one `.o.d` dependency file;
- NVCC command and architecture flags;
- cache directory layout;
- systemd OOM evidence;
- host-memory timeline;
- serial-build command;
- resulting shared object and its dynamic dependencies.

Answer:

1. Which component requested JIT compilation?
2. Why was `sm_120` or `sm_120f` selected?
3. How did generated translation units become Ninja jobs?
4. Which variable bounded job concurrency?
5. Why did host OOM occur after model load and `torch.compile` appeared done?
6. Which runtime libraries does the final `.so` require?
7. What changes cold startup, and what changes steady-state inference?

This becomes the Module 43 mastery artifact.

## Memory discipline

Before every full-model run estimate:

```text
resident weights
+ KV cache
+ activations/workspaces
+ CUDA graph pools
+ allocator fragmentation/headroom
= expected VRAM
```

Separately estimate host RAM:

```text
Python/runtime processes
+ checkpoint/loading buffers
+ generated-code compiler processes
+ object/linker memory
+ desktop/browser/IDE
= expected host pressure
```

VRAM utilization flags cannot protect host RAM from 16 concurrent NVCC jobs.

## Benchmark progression

For each stable server configuration:

1. compile and warm all measured paths;
2. record concurrency 1 for 128/128, 1024/256, and 8192/128;
3. sweep concurrency 2, 4, and 8;
4. sweep finite request rates through saturation;
5. record TTFT, TPOT/ITL, E2E, aggregate output throughput, and goodput;
6. log VRAM, clocks, power, temperature, and host pressure;
7. change one flag or implementation;
8. repeat in randomized order.

Do not optimize solely for maximum aggregate tokens/s if the intended workload
is an interactive agent.

## Profiler progression

Start small:

```text
PyTorch events/device timing
 → PyTorch profiler for operators
 → Nsight Systems for timeline
 → Nsight Compute for one kernel
 → PTX/SASS and resource inspection
```

Full vLLM Nsight Compute runs can be slow and noisy. Reproduce a kernel's shape
in the sandbox before collecting deep metrics.

## Hardware claims to avoid

- Do not generalize one RTX 5060 Ti result to all Blackwell GPUs.
- Do not copy B200 tile/resource assumptions without checking compute
  capability 12.0.
- Do not call emulated FP4 a native FP4 speed result.
- Do not report a forced fallback as an optimized quantized path.
- Do not infer multi-GPU scaling from single-GPU utilization.
- Do not compare engines with different prompts, output lengths, cache state, or
  quality settings.

The value of this GPU for the curriculum is not that it can run every model. It
is that it exposes the entire path from model metadata to native target code in
a system you can inspect completely.

