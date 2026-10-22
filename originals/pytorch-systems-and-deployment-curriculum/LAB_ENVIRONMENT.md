# Lab Environment and Reproducibility

The goal is not a maximal Python environment. It is a sequence of small,
reproducible environments whose failures can be explained. Keep this curriculum
repository as documentation and create the separate `torch-lab` repository for
code, locks, traces, and logs.

## Baseline inventory

Before changing a package, record:

- OS release, kernel, CPU, RAM, storage free space, and swap configuration;
- GPU name, VRAM, compute capability, driver, CUDA runtime, and toolkit;
- Python interpreter, PyTorch wheel or commit, CUDA/HIP build tag, Triton,
  compiler, CMake, Ninja, and `glibc` versions;
- package manager, environment path, relevant `PATH`, `LD_LIBRARY_PATH`,
  `CUDA_HOME`, compiler variables, and architecture-list variables;
- the exact command, model/checkpoint revision, input-shape distribution, and
  results directory.

In `torch-lab/environment/`, maintain `collect_env.sh`, `versions.lock.md`, and
`build-notes.md`. Run `python -m torch.utils.collect_env` as one input to the
manifest, but also capture the shell/toolchain context because it does not
encode every native-build decision.

## Separate environments by purpose

Use one environment at a time and document why it exists:

| Environment | Purpose | Rule |
|---|---|---|
| `torch-core` | Stable eager/compile/course experiments | Upgrade deliberately and record benchmark deltas |
| `torch-source` | PyTorch checkout/source build/tests | Keep separate from binary-wheel experiments |
| `torch-extension` | C++/CUDA/custom-op work | Pin compiler/toolkit/ABI details |
| `torch-deploy` | ONNX, TensorRT, AOTInductor experiments | Capture target runtime versions and providers |
| `torch-edge` | ExecuTorch/edge dependencies | Treat device/backend packages as target-specific |

Do not combine every CUDA package into one long-lived environment simply because
it happens to import today. That makes linker precedence and binary
compatibility failures impossible to localize.

## RTX 5060 Ti / Blackwell working contract

Your local GPU is Blackwell **SM120** with 16 GB VRAM and a CUDA 12.9 toolkit in
the current setup. Treat these as observations to verify at the start of a new
environment; driver, wheel, and extension support are moving targets.

For a tool that compiles GPU code at runtime, record all four independently:

1. the **hardware target** requested (for example an SM120 architecture list);
2. the **toolkit/compiler** that can emit that target;
3. the **binary/runtime libraries** chosen at load time; and
4. the **host resources** available during the build.

`nvidia-smi` reports device-side state. It does **not** prove that host RAM,
compiler processes, temporary-disk space, ABI, or dynamic-library ordering are
healthy.

## The FlashInfer build incident as a reusable case study

The earlier NVFP4 JIT compile failed because parallel compilation exhausted host
resources. The useful model is:

```text
peak host RAM ≈ concurrent compile units
                × (frontend + template instantiation + ptxas + linker memory)
                + Python/runtime overhead
```

Two controls can multiply one another:

- `MAX_JOBS=1` limits **outer** build/Ninja parallel work in tooling that honors
  it.
- `FLASHINFER_NVCC_THREADS=1` limits FlashInfer's **inner** `nvcc` thread/work
  strategy.

They solve host-memory pressure, not a GPU out-of-memory condition. A strong
postmortem contains the command, process count, RSS/swap evidence, source/cache
path, architecture flags, compiler output, and the smallest resource-limit
change that recovers the build. Module 40 generalizes this to PyTorch extensions
and source builds.

## Build hygiene

- Start native work with `MAX_JOBS=1` on a constrained host, then raise it one
  step at a time while measuring wall time and peak RSS. More jobs are not
  automatically faster.
- Keep build directories and compiler logs. They are evidence, not clutter.
- Never assume `CUDA_HOME` decides every library used at runtime. Inspect the
  build command and actual loader path when a CUDA extension behaves oddly.
- Avoid copying a broad `LD_LIBRARY_PATH` across all environments. Put any
  required library-path adjustment in a named activation script and record it.
- Keep architecture flags specific to the target. A successful generic build
  may still JIT, fall back, or fail on SM120.
- Distinguish **compile**, **link**, **load**, **first JIT execution**, and
  **steady-state execution**. They have different failure surfaces.
- Change exactly one compatibility variable at a time: wheel, toolkit,
  compiler, architecture list, extension commit, or environment path.

## Measurement hygiene

For GPU timing, warm up, synchronize at the appropriate boundary, capture a
shape distribution rather than one input, and report both host and device
metrics. For compilation, separately report cold build/compile time, cache-hit
time, generated artifact size, peak host RAM, peak VRAM, and steady-state
execution time.

Every lab should have an eager CPU or small-GPU reference path. This is the
oracle for compiler, custom-op, quantized, and exported variants.

## Safety and cleanup

Build caches can become large. Before deleting one, write down its owner,
version/hash, and whether it contains the only useful build log. Prefer moving
an identified stale cache aside or using the project’s documented cache-clean
command rather than deleting a broad directory. Never delete an unresolved
cache root as a troubleshooting reflex.
