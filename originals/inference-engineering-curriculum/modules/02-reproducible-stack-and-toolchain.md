# Module 02 — Reproducible Stack and Toolchain

> **Phase:** 0 — Orientation  
> **Time:** Week 1; two 3-hour sessions plus a 2–4 hour independent audit  
> **Prerequisites:** Module 01, shell basics, virtual environments, and Git

## Why this matters

GPU inference is a compatibility graph, not a single package install. A Python wheel can import yet fail when a JIT kernel targets the wrong streaming-multiprocessor (SM) feature set; a driver can run one CUDA runtime but not another; a stale extension cache can survive a package upgrade; two “same version” runs can select different kernels. Reproducibility means preserving the executable stack and the benchmark protocol—not merely saving `requirements.txt`.

## Outcomes

You will be able to:

- distinguish GPU compute capability, kernel driver, CUDA driver API, runtime, toolkit, PTX, and cubin;
- explain backward, minor-version, and restricted forward compatibility;
- lock Python and native dependencies and record wheel provenance;
- build a fail-fast environment probe and deterministic correctness smoke test;
- separate numerical reproducibility from performance repeatability.

## Mental model: a resolved compatibility graph

Read the stack bottom-up:

`GPU/SM ← kernel driver ← user-mode CUDA libraries ← framework wheel ← native extension/JIT cache ← Python package ← model`.

`nvidia-smi` reports the driver and a CUDA capability associated with that driver; it does **not** report the toolkit used by `nvcc`. `torch.version.cuda` describes the CUDA family against which that PyTorch build was produced. `nvcc --version` describes a local compiler toolkit. All three can differ legitimately, but the relationship must be understood and recorded.

From SM 9.0 onward, CUDA distinguishes baseline, family-specific, and architecture-specific features. For example, `compute_120`, `compute_120f`, and `compute_120a` are not synonyms. A binary using a family- or architecture-specific Tensor Core operation has a narrower compatibility set.

## Theory and executable contracts

Treat an environment as a tuple:

\[
E=(\text{GPU PCI ID},\text{SM},\text{driver},\text{OS/glibc},
\text{Python},\text{Torch build},\text{CUDA libs},\text{engine commit},
\text{kernel packages},\text{flags},\text{cache key}).
\]

A benchmark is reproducible only relative to \(E\), model/checkpoint hash, input distribution, and measurement method.

Your smoke tensor contract is deliberately small:

- \(A:[M,K]\), \(B:[K,N]\), \(C=A B:[M,N]\);
- compare CPU FP64 reference, CPU FP32, and GPU framework output;
- exercise a shape not divisible by common tile sizes, such as `[37, 113] @ [113, 29]`;
- synchronize before timing and test finite values plus a tolerance derived from dtype.

Determinism is not transitive across releases or devices. PyTorch explicitly does not promise bitwise identity across versions and platforms. Use deterministic modes for debugging, then benchmark the production kernel policy separately.

## Implementation ladder

1. Initialize `inference-lab/` with the prescribed directories, `pyproject.toml`, a checked-in lockfile, and `.gitignore` rules for generated caches and model weights.
2. Write `engine/env_report.py` to emit JSON containing `uname`, Python, package versions, wheel locations, Torch build config, GPU name/UUID/SM, driver, visible devices, `nvcc`, relevant environment variables, and Git commit/dirty state.
3. Add `tests/test_environment.py` with declared minimums and compatibility assertions. Error messages must show detected and required values.
4. Add `reference/smoke_matmul.py` and compare device output against FP64 for several dtypes and awkward shapes.
5. Save benchmark metadata beside results, never in a separate notebook that can drift.
6. Document cache locations for TorchInductor, Triton, CUDA JIT, vLLM, and FlashInfer. Invalidation must be explicit and scoped.

Do not copy a live environment’s entire `LD_LIBRARY_PATH` into documentation. Resolve why each library directory is needed.

## Two 3-hour sessions

**Session A — resolve the stack**

- **0:00–0:35 — Stack anatomy:** classify driver, runtime, toolkit, wheel, extension, PTX, and cubin.
- **0:35–1:05 — Compatibility cases:** reason through three driver/runtime/SM combinations using NVIDIA’s tables.
- **1:05–1:15 — Break.**
- **1:15–1:55 — Repository lab:** lock the project and implement `env_report.py`.
- **1:55–2:30 — Smoke test:** run awkward-shape matmul on CPU and GPU; record tolerances.
- **2:30–3:00 — Cold/warm lab:** measure first import/first kernel versus cached execution and locate generated artifacts.

**Session B — reproduce and break it safely**

- **0:00–0:45:** rebuild from the lockfile in a fresh environment and diff manifests.
- **0:45–1:30:** inspect loaded shared objects and relate them to wheel metadata.
- **1:30–2:15:** diagnose prepared ABI, SM-target, and stale-cache failures read-only.
- **2:15–3:00:** finalize the rebuild runbook and deterministic versus performance modes.

## Experiments and deliverables

Submit `uv.lock` (or an equivalently exact lock), environment report, smoke test, compatibility assertions, and `notes/reproduction.md`. Run the smoke test twice in one process, once in a new process, and once after changing only a cache directory. Explain compilation versus execution time.

Create a table with “source package version,” “binary CUDA dependency,” “loaded shared-object path,” and “SM target.” A package name alone is insufficient evidence.

## Failure modes

- Reading the “CUDA Version” in `nvidia-smi` as the installed toolkit.
- Mixing wheels from incompatible package indexes during unconstrained resolution.
- Pinning Python packages but not Git commits, model revisions, drivers, or flags.
- Globally deleting caches to fix one stale artifact.
- Benchmarking the first JIT run as steady-state latency.
- Setting deterministic algorithms and assuming performance remains representative.
- Loading multiple CUDA runtime families accidentally through broad library-path discovery.

## Mastery gate

Given a missing-symbol, “no kernel image,” or unsupported-architecture error, classify the failing edge in the compatibility graph and propose a read-only verification before reinstalling anything. Produce a fresh environment from the lockfile whose report matches all declared constraints and whose smoke outputs pass.

## Primary sources

- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [CUDA compute capabilities and feature targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [FlashInfer installation and binary/JIT packages](https://docs.flashinfer.ai/installation.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness)
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)

## Code paths to read

- vLLM: `setup.py`, `cmake/`, and `vllm/envs.py`
- FlashInfer: `flashinfer/jit/env.py` and `flashinfer/jit/cpp_ext.py`
- PyTorch: `torch/utils/cpp_extension.py`
- Your lab: `engine/env_report.py`, `tests/test_environment.py`

## OSS bridge

Run `flashinfer show-config`, `python -m torch.utils.collect_env`, and the engine’s version command, then compare their fields to your neutral report. Your report should remain useful if vLLM is replaced by TensorRT-LLM or llama.cpp.

## Next

Module 03 establishes shape algebra, the language used to verify every model and kernel boundary that follows.
