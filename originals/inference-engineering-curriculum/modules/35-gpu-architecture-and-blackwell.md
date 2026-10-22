# Module 35 — GPU Architecture and Blackwell

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 18, paired with Module 36; three 3-hour sessions and a hardware dossier  
**Prerequisites:** Modules 02, 05, 25–28, and 34; reproducible CUDA environments, roofline analysis, low-precision formats, and engine dispatch

## Why this matters

“Blackwell support” is not one capability bit. A kernel must agree with the exact compute capability, instruction family, toolkit, binary target, data layout, and GPU resource limits. The learner's successful FlashInfer build on an RTX 5060 Ti proves that this consumer GPU can compile and run a particular SM120 path; it does not prove that an SM100 datacenter cubin, every Blackwell Tensor Core instruction, or a benchmark from B200 transfers unchanged.

## Outcomes

You can:

- map grids, blocks, warps, lanes, and streaming multiprocessors to an inference kernel;
- distinguish architecture family, compute capability, PTX target, and native SASS;
- explain registers, shared/L1, L2, and device-memory movement in a roofline model;
- calculate occupancy constraints without treating maximum occupancy as the objective;
- produce a verified capability dossier for the RTX 5060 Ti and compare it with datacenter Blackwell.

## Mental model: a compatibility and resource contract

A CUDA grid contains thread blocks; blocks are scheduled on streaming multiprocessors (SMs); threads execute in 32-thread warps under SIMT. Divergent lanes serialize paths, while inactive lanes still consume an issued warp slot. A block stays on one SM and consumes registers, shared memory, thread slots, and a block slot until completion.

If a kernel uses \(R_t\) registers per thread, \(T_b\) threads and \(S_b\) bytes of shared memory per block, resident blocks per SM are bounded by

\[
B_{\mathrm{res}}\le
\min\left(
B_{\mathrm{arch}},
\left\lfloor\frac{T_{\mathrm{SM}}}{T_b}\right\rfloor,
\left\lfloor\frac{R_{\mathrm{SM}}}{R_tT_b}\right\rfloor,
\left\lfloor\frac{S_{\mathrm{SM}}}{S_b}\right\rfloor
\right),
\]

with allocation granularity and other architectural limits applied in practice. Occupancy is active warps divided by the architectural maximum. More resident warps can hide latency, but fewer warps with better locality, instruction-level parallelism, or Tensor Core utilization may run faster. Measure both occupancy and elapsed time.

The memory path is not a single “VRAM bandwidth” number. Threads hold values in registers; blocks explicitly share data through shared memory, which shares physical resources with L1 on current NVIDIA designs; L2 is chip-wide; device DRAM is largest and slowest. Coalescing, cache reuse, bank conflicts, and bytes moved at each level determine the applicable roofline:

\[
P\le\min(P_{\text{compute}}, I_{\text{level}}B_{\text{level}}).
\]

Hierarchical rooflines can expose an L1/shared or L2 ceiling even when DRAM bandwidth looks underused.

## CC, PTX, SASS, and this machine

Compute capability (CC) identifies programmable hardware features. NVIDIA's official table lists GeForce RTX 5060 Ti as **CC 12.0**. B200/GB200 are **CC 10.0**, while B300/GB300 are 10.3. They share the Blackwell name but differ in SM target, memory capacity/bandwidth, interconnect, deployment envelope, and supported binaries. Never substitute a B200 result for local evidence.

CUDA source is lowered through PTX, a virtual ISA, and assembled into native machine code (SASS) stored in cubins/fatbins. `-gencode` can embed native `sm_120` code, PTX such as `compute_120`, or both. Native SASS starts without driver JIT but is tied to compatible binary targets. Embedded PTX may be JIT-compiled by the driver when compatibility rules allow. Family-specific architecture targets and instructions—spelled with an `f` suffix in relevant CUDA/PTX tooling—have stricter scope than broadly portable PTX.

Reconnect this to the FlashInfer exercise: `CUDA_HOME` selected the compiler, the architecture list selected SM120/family code generation, JIT produced a cached shared object, and the driver loaded its native code. Inspecting `nvidia-smi` alone cannot validate that chain. Record `nvcc --version`, framework CUDA build, GPU CC, fatbin targets, JIT logs, and the kernel actually dispatched.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/gpu_capability.py` to record GPU name, CC, driver, toolkit, framework runtime, memory, and supported framework arch list. Add a worksheet that predicts whether source, PTX-only, and cubin-only artifacts should load.

### 2. Optimized

Implement `inference-lab/kernels/cuda/resource_probe.cu`: vector add, divergent branch, coalesced/strided load, and configurable register/shared-memory kernels. Compile for explicit PTX and SASS targets; inspect with `cuobjdump` and `nvdisasm`.

### 3. Integrated

Trace one FlashInfer NVFP4 operator from Python dispatch through JIT cache and shared object to SM120 instructions. Compare profiler counters and resource usage with a fallback path. State exactly what the trace proves and what remains library- or datacenter-specific.

## Three 3-hour sessions

1. **Architecture map:** draw grid-to-SM execution and memory hierarchy; capture the local capability dossier and a B200 contrast table.
2. **Binary inspection:** compile fatbins, list embedded targets/PTX, disassemble SASS, and deliberately test an unsupported native target.
3. **Resource experiment:** sweep block size, registers, shared memory, stride, and divergence; place results on hierarchical rooflines.

## Experiments and deliverables

Commit:

- `reference/gpu_capability.py`;
- `kernels/cuda/resource_probe.cu`;
- `tests/test_binary_targets.py`;
- `benchmarks/results/sm120-resource-sweep.json`;
- `notes/hardware/rtx5060ti-sm120.md`;
- `notes/source_walks/flashinfer-sm120-dispatch.md`.

Record compiler command, `-gencode`, PTX ISA, SASS target, register/shared usage, theoretical and achieved occupancy, clocks/power state, cache behavior, warm-up, and raw timings.

## Common failure modes

- Equating “Blackwell” with one compute capability.
- Reading the driver's maximum CUDA compatibility as the toolkit used to build.
- Assuming PTX and SASS are interchangeable artifacts.
- Omitting PTX and discovering a cubin lacks the deployed SM.
- Assuming PTX JIT grants instructions unavailable to its target.
- Maximizing occupancy while increasing spills or destroying tiling.
- Using peak marketing TOPS as a measured kernel ceiling.
- Claiming native FP4 execution without inspecting dispatch and generated code.

## Mastery gate

Given a `.so` or fatbin, identify its embedded targets, predict whether it loads on SM120, and verify with tools. For one kernel, predict the limiting resident-block resource, compare with profiler occupancy, and explain why its fastest configuration does or does not have maximum occupancy. Present a sourced CC12.0-versus-CC10.0 table without transferring datacenter claims to the RTX 5060 Ti.

## Canonical code areas and OSS bridge

Study CUDA platform checks and kernel registries in [vLLM](https://github.com/vllm-project/vllm), JIT/environment and architecture handling in [FlashInfer](https://github.com/flashinfer-ai/flashinfer), and architecture layouts/instructions in [CUTLASS](https://github.com/NVIDIA/cutlass). Useful contributions include accurate SM capability guards, architecture-list tests, readable unsupported-target errors, and local SM120 benchmark data.

## Primary and authoritative sources

- [NVIDIA CUDA GPU compute-capability table](https://developer.nvidia.com/cuda-gpus)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)

## Next

Module 36 turns the static hardware map into correct asynchronous execution, memory access, allocation, and synchronization experiments.
