# Lab Environment and Reproducibility

## The principle: separate concerns before debugging them

An inference process crosses several independently versioned layers:

```text
model/config/checkpoint
  → Python model and engine
  → PyTorch/JAX runtime and dispatcher
  → generated or prebuilt kernel library
  → CUDA runtime + math libraries
  → CUDA driver
  → GPU ISA and hardware
```

`nvcc --version`, `torch.version.cuda`, and the maximum CUDA version printed by
`nvidia-smi` describe different layers. They do not need identical text, but
compiled components must obey their documented compatibility and ABI
requirements.

The FlashInfer `sm_120` episode is a model lab:

- the model and GPU were compatible;
- the optimized kernel was generated just in time;
- Ninja scheduled many expensive NVCC translation units concurrently;
- host memory pressure, not GPU OOM, killed the process;
- `MAX_JOBS=1` changed build concurrency, not inference throughput;
- the build directory and `build.ninja` exposed the actual commands and targets.

Keep that incident as the first `failure_postmortems/` entry.

## Use three environments

Do not turn one virtual environment into every experiment.

### 1. Stable serving baseline

Purpose: reproduce known inference results. Pin:

- Python minor version;
- PyTorch build including CUDA suffix;
- engine version or commit;
- FlashInfer/CUTLASS/Triton versions;
- tokenizer and model revisions.

Change this environment only through a reviewed lock-file update.

### 2. Runtime development environment

Purpose: editable vLLM, SGLang, or related source builds. Follow the target
project's current build instructions. Keep one target repository per
environment when compiled dependencies conflict.

### 3. Kernel sandbox

Purpose: small CUDA, Triton, and CUTLASS programs with short build times and
unambiguous profiler traces. Do not require a full model server to test a
softmax, reduction, dequantizer, or GEMM.

For each environment, record why it exists and what must not be installed into
it.

## Baseline repository setup

Create the cumulative lab outside this curriculum:

```bash
mkdir inference-lab
cd inference-lab
git init
```

Use `uv`, `venv`, Conda, or containers consistently; the tool matters less than
capturing exact resolution. For Python environments save:

```bash
python -VV
python -m pip freeze
python -m pip check
```

For `uv`, commit `pyproject.toml` and `uv.lock`. Avoid unconstrained installation
commands in final reports.

## Environment manifest

Every benchmark report should link a text or JSON manifest containing:

```bash
uname -a
lscpu
free -h
df -h
nvidia-smi
nvidia-smi -q
nvcc --version
cmake --version
ninja --version
gcc --version
python -VV
python -m torch.utils.collect_env
```

Also record:

- exact GPU model, VRAM, power limit, and compute capability;
- driver version;
- CUDA toolkit path and version;
- framework CUDA build version and supported architecture list;
- engine, compiler, and kernel-library Git commits;
- CPU governor and GPU power/clock policy if changed;
- model and tokenizer repository revisions;
- relevant environment variables;
- cache state: cold, warm, or rebuilt;
- host processes that materially compete for CPU or memory.

Redact hostnames, tokens, private registry paths, and usernames before attaching
manifests to public issues.

## Build-tool minimum

Become comfortable with:

- a C++20-capable host compiler supported by the chosen CUDA toolkit;
- `nvcc` and its host/device compilation phases;
- CMake configure versus build phases;
- Ninja targets, dependency files, job pools, and `-j`;
- `readelf`, `objdump`, `nm`, `ldd`, and loader search rules;
- `cuobjdump` and `nvdisasm`;
- `compute-sanitizer`;
- Nsight Systems (`nsys`) and Nsight Compute (`ncu`);
- Git worktrees, `git bisect`, and patch-based review.

Do not install tools just to check a box. Each enters the course at the module
where it answers a concrete question.

## GPU target policy

Use the RTX 5060 Ti as the primary optimization target:

- Blackwell, compute capability 12.0;
- enough memory for focused kernels and selected compressed models;
- useful native low-precision behavior;
- limited VRAM, which makes cache and scheduler trade-offs visible.

Maintain two testing modes:

1. **Native target:** real `sm_120` compilation and profiler measurements.
2. **Portable correctness:** CPU/PyTorch references and CI-friendly small
   shapes that do not require that GPU.

Distributed modules can begin with multiple local processes and analytical
communication models. Real multi-GPU claims require access to real multi-GPU
hardware; never present a simulation as a throughput result.

## Cache discipline

Know the location and semantics of:

- Hugging Face model and dataset caches;
- PyTorch extension caches;
- TorchInductor/Triton caches;
- FlashInfer JIT caches;
- CMake and Ninja build trees;
- compiler caches such as `ccache`, if used.

For each experiment label the run:

- **cold build:** generated code absent;
- **warm build:** binary exists, runtime cold;
- **warm runtime:** kernels, graphs, and model pages exercised.

Never include compilation time in request latency unless the product
requirement explicitly concerns cold start.

Do not solve cache problems by recursively deleting broad directories. Inspect
targets and manifests first, preserve failing artifacts for diagnosis, and
remove only an explicit cache subtree when rebuilding is part of the
experiment.

## Correctness baseline

Every optimized component needs:

- a slow reference implementation;
- shape, dtype, stride, alignment, and device test matrices;
- edge cases such as zero-length, tails, odd dimensions, non-contiguous input,
  NaN/Inf, and extreme values where semantically valid;
- dtype-aware absolute/relative error thresholds;
- randomized/property tests with reproducible seeds;
- sanitizer runs for CUDA code;
- generation-level checks when the operator changes a model.

Performance tests and correctness tests are separate. A benchmark is not a
test, and passing `allclose` once is not a numerical analysis.

## Recommended source checkouts

Clone only as each module needs it:

- [PyTorch](https://github.com/pytorch/pytorch)
- [Triton](https://github.com/triton-lang/triton)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp/ggml](https://github.com/ggml-org/llama.cpp)

Pin a commit in every source-walk note. Repository `main` branches are moving
targets.

## Disk and memory budget

Before creating another large environment, estimate:

```text
model checkpoints
+ Python/CUDA wheels
+ source trees
+ object files and generated kernels
+ profiler reports
+ duplicate caches
+ temporary link space
```

Keep at least enough free space for the largest build plus its temporary
objects. Track host RAM and swap during CUDA compilation as carefully as VRAM
during serving.

## Authoritative setup references

- [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [PyTorch C++/CUDA custom operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [Ninja manual](https://ninja-build.org/manual.html)
- [CMake documentation](https://cmake.org/documentation/)

