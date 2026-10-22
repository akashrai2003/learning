# Module 40 — Build, JIT, Ninja, CMake, and ABI Forensics

**Phase:** 6 — Native PyTorch and GPU integration  
**Time:** 15 focused hours  
**Prerequisites:** Modules 36–39, shell literacy, C++/CUDA compilation concepts, and a virtual environment.

## Existing inference curriculum bridge

This is the direct PyTorch-side continuation of [02 — Reproducible stack/toolchain](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), [35 — Blackwell GPU](../../inference-engineering-curriculum/modules/35-gpu-architecture-and-blackwell.md), [41 — Custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [42 — Compiler stack](../../inference-engineering-curriculum/modules/42-compiler-stack-dynamo-inductor-and-triton.md), and especially [43 — Build, JIT, Ninja, CMake, and ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md).

## Why this matters and the FlashInfer SM120 incident

The successful NVFP4 launch was a real compiler-systems lesson. vLLM had loaded the Gemma checkpoint and logged that `torch.compile` completed. FlashInfer then JIT-generated its SM120 NVFP4 CUTLASS operation. The cache’s `build.ninja` exposed many independent CUDA translation-unit edges; Ninja launched them in parallel, host memory pressure rose, and `systemd-oomd` killed the session. That was a host-side build-concurrency failure, not a VRAM OOM, missing Blackwell support, or an inference-kernel correctness failure.

You will reconstruct this whole chain, control it deliberately, and diagnose compile, link, import, driver-JIT, and launch failures with artifacts rather than folklore.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P40 — Build, JIT, Ninja, CMake, and ABI**.

1. [PyTorch from-source build](https://github.com/pytorch/pytorch#from-source) — learn the API or system contract.
2. [C++ extension docs](https://docs.pytorch.org/docs/stable/cpp_extension.html) — study the underlying design, paper, or specification.
3. [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch custom-ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch extension-cpp examples](https://github.com/pytorch/extension-cpp) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: staged artifacts and two kinds of parallelism

```text
sources + headers + flags + architecture target
  → configure/JIT cache key → build.ninja dependency DAG
  → C++/CUDA object files → device/host linking → .so
  → dynamic loader resolves libraries → driver loads cubin or JITs PTX
  → kernel launches on a stream
```

Ninja’s job count controls how many independent *build edges* run at once. In PyTorch’s C++ extension tooling, `MAX_JOBS` caps its Ninja worker invocation (the documented default may be CPU count plus two). FlashInfer’s `FLASHINFER_NVCC_THREADS` is a project-specific control that feeds NVCC’s internal `--threads` behavior for one compile edge. They are not interchangeable and neither is an inference throughput flag. Peak host build pressure is roughly concurrent translation units × compiler process memory × inner compiler parallelism. On the affected host, serializing both dimensions—`MAX_JOBS=1`, `FLASHINFER_NVCC_THREADS=1`, or direct `ninja -j1` for the existing directory—traded one-time build time for a successful result; warm generated kernels execute the same machine code.

Treat every environment variable as implementation-specific until you verify it in the pinned source and generated command. `MAX_JOBS` does not universally affect arbitrary CMake/Ninja invocations. Conversely, direct `ninja -j1` controls outer scheduling but says nothing about an NVCC `--threads` argument already written into `build.ninja`. Capture both values in build evidence.

## Toolchain, architecture, and ABI contracts

`CUDA_HOME` chooses a compiler toolkit; `nvidia-smi`’s CUDA version reports driver capability, not the toolkit selected by a build or the runtime loaded by Python. For Blackwell, the target needs compatible `sm_120` code or PTX that the installed driver can JIT. Inspect the actual `nvcc` command, architecture flags, fatbin/cubin/PTX contents, and device capability—not only an environment variable such as `FLASHINFER_CUDA_ARCH_LIST=12.0`.

An extension must also align CPython, PyTorch/libtorch, host compiler/libstdc++ ABI (including relevant `_GLIBCXX_USE_CXX11_ABI`), CUDA runtime SONAMEs/symbols, and driver. A binary can link yet fail during import or first use. Use `readelf -d` for `NEEDED`/RPATH/RUNPATH, `ldd` or loader diagnostics for actual resolution, `nm -D` plus `c++filt` for symbols, and CUDA binary inspection for embedded targets. A broad `LD_LIBRARY_PATH` can accidentally load CUDA 12 and CUDA 13 runtimes into one process; prefer a matched wheel/environment and narrow, documented paths.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/build/artifact_audit.sh` to report Python/PyTorch ABI config, compiler/CMake/Ninja/NVCC versions, device/driver/toolkit distinction, build command, environment allowlist, ELF dependencies, and embedded targets for one extension.
2. **Optimized:** build the Module 36 op with both PyTorch extension tooling and a minimal CMake→Ninja project. Sweep outer jobs and inner NVCC threads separately, record wall time/peak host RSS, and hash artifacts to prove whether the outputs changed.
3. **Integrated:** preserve the FlashInfer cache’s generated `build.ninja` and create `torch-lab/build/flashinfer_sm120_postmortem.md`. Reproduce four controlled failures: unsupported target, missing shared library, stale framework cache, and host OOM pressure. Rebuild cleanly and import/run a correctness test.

## Three focused sessions

1. **Build graph:** annotate every compile/link edge and cache-key input in a generated Ninja file. Explain why a JIT directory is diagnostic evidence, not clutter.
2. **Resource experiment:** vary only outer jobs versus inner NVCC threads; log RSS, swap/PSI/cgroup or OOM evidence, CPU utilization, output hashes, build time, and warm runtime.
3. **ABI forensic drill:** use ELF/CUDA tools on good and intentionally mismatched `.so` files; resolve one failure with a matched environment rather than a giant library-path workaround.

## Experiments and deliverables

Commit `artifact_audit.sh`, `build_matrix.json`, `CMakeLists.txt`, `test_clean_import.py`, `flashinfer_sm120_postmortem.md`, and `abi_evidence/`. Record source commit, Python/PyTorch/CUDA/driver/compiler/CMake/Ninja versions, architecture flags, outer/inner concurrency, peak host RSS, build time, cache key/state, artifact hash, `DT_NEEDED`, resolved paths, symbols, embedded targets, and exit cause.

## Common failure modes

- Calling a host OOM a GPU OOM because a CUDA project was running.
- Multiplying Ninja jobs by unlimited NVCC threads.
- Deleting the failing cache before saving `build.ninja` and the exact command.
- Assuming `MAX_JOBS` changes warm token throughput.
- Trusting driver capability as proof of compiler/runtime/ABI compatibility.
- Fixing an import by adding every `site-packages/nvidia/*/lib` directory to `LD_LIBRARY_PATH`.

## Mastery gate

Given a failed JIT cache, reconstruct and rerun the exact edge, classify it as compiler/link/loader/driver/launch/host-resource failure, and make the smallest safe correction. Show why serialized JIT compilation fixes peak host memory without changing the warm kernel’s output or throughput, then audit the final `.so` in a clean environment.

## Primary sources, code paths, and OSS bridge

- [PyTorch C++ extension and `MAX_JOBS` documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [NVCC compiler-driver documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [CMake build-system manual](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html)
- [Ninja manual](https://ninja-build.org/manual.html)
- [FlashInfer source](https://github.com/flashinfer-ai/flashinfer)

Read `torch/utils/cpp_extension.py`, FlashInfer’s pinned `flashinfer/jit/` build/environment code, generated cache files, and vLLM’s [incremental build guide](https://docs.vllm.ai/en/latest/contributing/incremental_build/). Useful contributions are bounded-job defaults, cache-key/diagnostic fixes, ABI/architecture error clarity, or exact postmortems. Preserve correctness and a clean rebuild path before optimizing build time.

## Next

Module 41 covers CUDA Graph replay and allocator behavior once an operator/model is successfully built.
