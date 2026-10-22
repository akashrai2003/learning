# Capstone 03 — Kernel Optimization Suite

## Objective

Take three model-relevant operations from a correct PyTorch reference to
profiled GPU implementations, then integrate at least one as a real custom
operator with dispatch and fallback.

Recommended operations:

1. RMSNorm or fused residual + RMSNorm;
2. online softmax or exact tiled attention;
3. grouped low-bit dequantization fused with linear/GEMM or its epilogue.

The suite should include both Triton and CUDA/CUTLASS work so you learn where
their abstraction boundaries differ.

## Optimization loop

For every kernel:

1. define semantic and dtype/layout contracts;
2. retain a slow, high-precision oracle;
3. enumerate real model shapes;
4. estimate FLOPs, bytes, and likely bottleneck;
5. implement the simplest correct GPU version;
6. profile system timeline, then kernel metrics;
7. change one mechanism;
8. rerun correctness and performance;
9. document the new bottleneck;
10. establish dispatch/fallback boundaries.

## Shape suites

Use shapes representing:

- prefill: large \(M\), model hidden/intermediate \(K,N\);
- decode: tiny \(M\), large weight matrices;
- non-multiples and tail tiles;
- GQA head counts and dimensions;
- quantization group boundaries;
- model-specific irregular dimensions.

Do not report only the shape that matches the selected tile perfectly.

## Required implementation progression

### Triton

- explicit program IDs and masked pointer operations;
- fused reduction/normalization;
- tiled matmul or attention;
- autotune configurations with a justified search space;
- generated PTX/resource inspection.

### CUDA C++

- grid/block indexing and bounds;
- coalesced/global and shared-memory tiled versions;
- warp reductions;
- asynchronous/pipelined movement where supported and justified;
- stream-safe launches and explicit error checking.

### CUTLASS/CuTe

- build and run an official example;
- explain its tensor/layout and collective concepts;
- change tile, dtype, epilogue, or block-scale configuration;
- map template configuration to generated target architecture code.

### PyTorch integration

- dispatcher registration;
- CPU or composite reference/fallback;
- CUDA implementation;
- FakeTensor/meta behavior for `torch.compile`;
- `torch.library.opcheck` and numerical tests;
- wheel or reproducible extension build;
- clear architecture and dtype errors.

## Measurements

Report:

- device time distribution;
- achieved GB/s and/or TFLOP/s;
- percentage of an appropriate measured or documented roof;
- launch count and CPU gaps;
- registers, shared memory, active warps, and actual limiter;
- relevant memory transactions/cache behavior;
- compilation time and cache size separately;
- end-to-end block/model change.

An occupancy increase is not itself a success criterion.

## JIT/build exercise

Recreate a controlled target-architecture build:

- vary architecture list;
- inspect generated Ninja rules and NVCC commands;
- cap Ninja and compiler concurrency;
- identify object, fatbin/cubin, shared library, and cache artifacts;
- use `readelf`/`ldd`/`nm` to inspect dependencies and symbols;
- intentionally introduce a missing architecture or ABI mismatch and write the
  diagnostic procedure.

Do not destabilize the stable serving environment; use the kernel sandbox.

## Acceptance gate

For one kernel, a reviewer must be able to:

- reproduce build and measurements;
- see tests fail under an intentionally broken implementation;
- understand supported and rejected shapes;
- trace Python call to registered operator to launched kernel;
- verify the speedup over a strong baseline across the declared dispatch range;
- see an honest end-to-end effect or explanation of why it is absent.

## Stretch work

- CUDA Graph capture safety;
- persistent kernels;
- warp specialization;
- Tensor Memory Accelerator or architecture-specific paths;
- FP8/FP4 Tensor Core implementation;
- FlashInfer or runtime-specific integration;
- multi-GPU collective overlap.

