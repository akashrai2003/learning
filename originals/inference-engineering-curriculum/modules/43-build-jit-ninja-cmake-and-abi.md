# Module 43 — Build, JIT, Ninja, CMake, and ABI

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 22, paired with Module 44; three 3-hour sessions and a reproduced build/load failure  
**Prerequisites:** Modules 02, 35, 41–42; CUDA architecture targets, custom operators, generated code, shells, and virtual environments

## Why this matters

Kernel JIT crosses Python, C++, CUDA, the build system, linker, dynamic loader, framework, and driver. A host killed by parallel compiler jobs is not a GPU OOM. A linked `.so` may still load the wrong CUDA runtime. A generated `build.ninja` is valuable evidence, not disposable clutter. Build speed controls such as `MAX_JOBS` affect time and peak host RAM; with identical output bytes they do not change steady-state token throughput.

## Outcomes

You can:

- read CMake-generated or JIT-generated Ninja rules and reconstruct the exact commands;
- control outer build jobs and inner NVCC threads without oversubscribing host RAM;
- distinguish compile, device link, host link, import/load, and launch failures;
- audit ELF dependencies, symbols, RPATH/RUNPATH, C++/Python/PyTorch/CUDA ABI, and GPU targets;
- preserve, invalidate, and reproduce JIT-cache artifacts deliberately.

## Mental model: a staged artifact pipeline

```text
sources + headers + options
 -> CMake/configuration or JIT key
 -> build.ninja dependency graph
 -> host/device object files
 -> device link and host link
 -> Python extension/shared object
 -> dynamic loader resolves DT_NEEDED symbols
 -> driver loads cubin or JITs PTX
 -> kernel launch
```

Ninja schedules independent build edges. PyTorch's extension backend normally emits `build.ninja`; its default worker count may exceed available memory, and `MAX_JOBS` caps that outer concurrency. FlashInfer separately exposes `FLASHINFER_NVCC_THREADS`, which controls the `nvcc --threads` value inside each compile edge. NVCC can also parallelize optimization work with `--split-compile`. Approximate peak pressure is the number of simultaneous translation units times their per-process compiler threads and template memory. More logical CPUs do not imply enough RAM.

On a host OOM kill, look for exit 137/SIGKILL, cgroup memory events, or kernel logs; CUDA allocator metrics may remain normal. Reduce `MAX_JOBS`, NVCC internal threads, architectures, or debug/LTO burden one variable at a time. Do not change mathematical kernel options while diagnosing resource pressure.

## ABI is several contracts

An importable extension must align at multiple boundaries:

- CPython extension ABI and Python version, unless using a supported limited/stable ABI;
- PyTorch dispatcher/libtorch ABI and operator schemas;
- host compiler/libstdc++ ABI, including relevant `_GLIBCXX_USE_CXX11_ABI`;
- CUDA runtime and dependent library SONAMEs/symbol versions;
- device code target and driver compatibility;
- kernel-level dtype, layout, and stream contract.

Use `readelf -d` for `NEEDED`, RPATH, and RUNPATH; `ldd` or loader diagnostics for resolved paths; `nm -D`/`c++filt` for exported or missing symbols; `cuobjdump` for embedded PTX/cubins. A broad `LD_LIBRARY_PATH` can hide a broken wheel combination and load two CUDA runtime generations into one process. Prefer matched packages and narrow, documented search paths.

The learner's successful SM120 FlashInfer build should become a reproducible case study: archive its generated Ninja file, exact `nvcc` line, `MAX_JOBS` and `FLASHINFER_NVCC_THREADS`, architecture target, resulting `.so` dependencies, cache key/location, and import trace. Verify the framework's CUDA build and extension requirements match; `nvidia-smi`'s “CUDA Version” is only driver capability.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/artifact_audit.sh` to report interpreters, compilers, PyTorch ABI/configuration, CUDA toolkit/runtime, dynamic dependencies, symbols, and device targets for one extension. Keep it read-only and machine-readable where practical.

### 2. Optimized

Build the Module 41 operator using both PyTorch JIT extension tooling and a minimal CMake-to-Ninja project under `inference-lab/kernels/custom_op/`. Vary only outer `MAX_JOBS` and inner `FLASHINFER_NVCC_THREADS`/NVCC thread controls; record wall time and peak host RSS, then hash the output artifacts.

### 3. Integrated

Inject and diagnose four failures: unsupported `sm` target, missing shared library, stale cache built against another framework, and host OOM from oversubscription. Rebuild in a clean environment, import without broad library-path overrides, and run one correctness test.

## Three 3-hour sessions

1. **Build graph:** configure CMake/Ninja and a PyTorch JIT build; annotate every compile/link edge in generated `build.ninja`.
2. **Resource and cache:** sweep outer/inner concurrency, capture host RAM/CPU, and inspect cache invalidation after source/flag/SM changes.
3. **ABI forensics:** use ELF/CUDA tools on good and broken `.so` files; write evidence-led postmortems and a clean rebuild recipe.

## Experiments and deliverables

Commit:

- `reference/artifact_audit.sh`;
- `kernels/custom_op/CMakeLists.txt`;
- `tests/test_extension_clean_import.py`;
- `benchmarks/results/build-concurrency.json`;
- `notes/builds/flashinfer-sm120-build.md`;
- `notes/failure_postmortems/extension-abi.md`.

Record source revision, Python/framework/toolkit/driver, host compiler, CMake/Ninja/NVCC versions, full command, environment diff, architecture list, jobs/threads, peak host RSS, build time, artifact hash, `DT_NEEDED`, resolved libraries, cache state, and process exit cause.

## Common failure modes

- Assuming the GPU caused a compiler process's host OOM.
- Multiplying Ninja jobs by unlimited NVCC threads.
- Deleting `build.ninja` before reading the actual failing command.
- Treating `MAX_JOBS` as a runtime inference-tuning flag.
- Changing job count and code-generation flags together.
- Trusting `nvidia-smi` as the selected compiler/runtime.
- Fixing import by adding every environment library directory to `LD_LIBRARY_PATH`.
- Mixing framework and extension wheels built for different CUDA ABIs.
- Reusing a JIT cache after compiler, framework, source, or SM changes.
- Calling successful linking proof that the kernel can launch.

## Mastery gate

Given a failing JIT directory, reconstruct and rerun the failed edge, identify whether termination was compiler error or host kill, and produce the smallest corrective change. Audit the final `.so` through dynamic dependencies and embedded GPU targets, then import/run it in a clean shell. Demonstrate that changing `MAX_JOBS` changes build resources but, for byte-identical output, not warm inference throughput.

## Canonical code areas and OSS bridge

Read PyTorch's `torch/utils/cpp_extension.py`, FlashInfer's JIT/build environment code, vLLM's `setup.py`, `CMakeLists.txt`, `cmake/`, and incremental-compilation guide, plus the generated files themselves. Strong contributions include cache-key fixes, bounded job defaults, clearer ABI/arch diagnostics, smaller reproductions, and documentation based on exact commands.

## Primary and authoritative sources

- [PyTorch C++ extension and `MAX_JOBS` documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [NVCC compiler-driver documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [CMake build-system manual](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html)
- [Ninja manual](https://ninja-build.org/manual.html)
- [vLLM incremental compilation workflow](https://docs.vllm.ai/en/latest/contributing/incremental_build/)

## Next

Module 44 converts these reproducible artifacts into trustworthy correctness, profiler, latency, throughput, and regression evidence.
