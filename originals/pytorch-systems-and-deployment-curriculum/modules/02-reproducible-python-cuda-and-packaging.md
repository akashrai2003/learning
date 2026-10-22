# Module 02 — Reproducible Python, CUDA, and Packaging

> **Phase:** I — Engineering foundations  
> **Time:** Week 1; three focused 3-hour sessions  
> **Prerequisites:** Module 01, shell fluency, Git, and a working Python environment

## Existing inference curriculum bridge

This is the PyTorch-native counterpart to [Module 02 — `reproducible-stack-and-toolchain.md`](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), [Module 35 — `gpu-architecture-and-blackwell.md`](../../inference-engineering-curriculum/modules/35-gpu-architecture-and-blackwell.md), and [Module 43 — `build-jit-ninja-cmake-and-abi.md`](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md). Reuse its RTX 5060 Ti SM120 dossier rather than guessing from `nvidia-smi`.

## Why this matters and outcomes

The most expensive PyTorch bugs often begin before `import torch`: a resolver picks a CPU wheel, a CUDA extension is compiled against different headers than the loaded Torch, or an old cache silently survives a toolchain change. “CUDA 12.9 installed” is not a sufficient description. You need an auditable compatibility graph and a small protocol for changing it.

You will be able to create locked Python projects, select and verify a PyTorch binary, distinguish driver, runtime, toolkit, PTX, and native SASS, build a small extension in a controlled environment, inspect its dynamic dependencies, and report a failure without leaking an ad-hoc `LD_LIBRARY_PATH` into every shell.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P02 — Reproducible Python, CUDA, and packaging**.

1. [PyTorch get-started matrix](https://pytorch.org/get-started/locally/) — learn the API or system contract.
2. [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) — study the underlying design, paper, or specification.
3. [PyTorch C++ extension docs](https://docs.pytorch.org/docs/stable/cpp_extension.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch developer notes](https://docs.pytorch.org/docs/stable/community/contribution_guide.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch source repository](https://github.com/pytorch/pytorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch developer discussion](https://dev-discuss.pytorch.org/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: compatibility is a graph, not a version string

Read the system bottom-up:

```text
GPU capability -> kernel driver -> CUDA driver API -> runtime libraries
  -> PyTorch wheel/libtorch -> extension headers/ABI -> Python package -> model code
```

`nvidia-smi` reports the installed driver and a driver-supported CUDA capability; it is not `nvcc --version`. `torch.version.cuda` describes the runtime family used to build that PyTorch binary. The toolkit provides `nvcc`, headers, and libraries for compiling extensions. The driver is backward compatible with older runtimes, but extensions need matching PyTorch, C++, CUDA, Python, and glibc contracts.

Record an environment tuple:

\[
E=(\text{OS/glibc},\ \text{Python},\ \text{resolver+lock},\ \text{Torch build},\ \text{CUDA runtime},\ \text{toolkit},\ \text{driver},\ \text{GPU/SM},\ \text{compiler},\ \text{extension flags}).
\]

Reproducibility means a known transformation from the lockfile and \(E\) to a test result—not that every machine has identical hardware.

## Packaging and native-build theory

Use `pyproject.toml` as the project boundary: runtime dependencies, optional `dev` dependencies, build backend, package metadata, and Python version constraint belong there. A lockfile records resolved artifacts and hashes; it is distinct from a loose list of desired packages. Keep packages importable from `src/`, then run tests against the installed package rather than the current directory by accident.

A native extension has two relevant build-time interfaces: the Python extension ABI and the C++/libtorch ABI. Its ELF `DT_NEEDED` entries are resolved by the dynamic loader through embedded `RPATH`/`RUNPATH`, standard locations, and environment search paths. `ldd`, `readelf -d`, and `nm -D` answer different questions: dependencies, loader metadata, and exported/undefined symbols. They do not prove a kernel has a matching architecture image; `cuobjdump`/`nvdisasm` and a runtime smoke test cover that layer.

For CUDA, compile targets should be intentional. A cubin targets a concrete SM; PTX is a virtual ISA that the driver may JIT for a compatible later device, subject to features and toolkit support. Do not conflate `TORCH_CUDA_ARCH_LIST`, a package's separate architecture flag, and a GPU's compute capability. Cache keys must include source, compiler/flags, Torch/CUDA versions, and architecture; otherwise a cache is a source of false confidence.

## Implementation ladder in the proposed `torch-lab/`

1. **Reference:** add `pyproject.toml`, a lockfile, `scripts/env_report.py`, and a `--smoke` command. The report prints Python, PyTorch, `torch.version.cuda`, `torch.__config__.show()`, driver/GPU information, and selected package versions without assuming a GPU exists.
2. **Optimized:** add `src/torch_lab/ops/add_one/` containing a tiny CPU extension built by the project’s declared backend. Exercise it through a test and record compiler commands. Then build a CUDA variant only after the CPU path is reproducible.
3. **Integrated:** add an environment manifest to every benchmark artifact, a cache-directory policy, and `scripts/audit_native_artifact.py` that displays `readelf -d`, `ldd`, target flags, and import-smoke output. The benchmark command refuses to compare results with incompatible manifests.

## Three focused sessions

1. **Session A — resolve and verify (3h):** make a fresh virtual environment with your chosen resolver; install the documented Torch build; save resolver/Python/Torch output; run CPU tensor and CUDA availability tests. Compare `nvidia-smi`, `nvcc --version`, and `torch.version.cuda` in writing.
2. **Session B — build anatomy (3h):** build the smallest extension, locate its `.so`, inspect `build.ninja` or build logs, run `readelf -d` and `ldd`, and alter one compile definition to prove cache invalidation works.
3. **Session C — failure drills (3h):** diagnose a deliberate missing shared library, wrong arch list, and stale build directory. For each, identify whether failure is at compile, link, import, driver JIT, or launch; restore the clean lockfile afterward.

## Experiments and deliverables

Write `docs/runs/0002-environment-dossier.md`. Include command lines, wheel URLs/hashes or lock entries, toolkit and driver data, GPU PCI/SM identity, extension flags, cache locations, and a clear distinction between cold compilation and warm execution. Make a matrix test that at least covers CPU and your local CUDA target. Package only the toy Python code initially; extension distribution is a later lesson.

## Common failure modes

- Installing a second CUDA runtime globally to satisfy one extension and thereby loading two incompatible runtimes in one process.
- Treating a successful `import torch` as proof that a generated CUDA kernel targets SM120.
- Mixing editable source, a global `pip install`, and a cached JIT binary without knowing which one imported.
- Reporting a copied shell command without resolver version, wheel provenance, lock state, and compiler command.
- Deleting all caches as the first diagnostic action and erasing the evidence needed to reproduce a build bug.

## Mastery gate

From a new environment, install the project from its lock, run CPU and GPU smoke tests, build the tiny extension, and produce an artifact audit. Given a deliberately mismatched extension, classify the broken contract and show the exact evidence—not merely a reinstall. File a local issue report with a minimal command and environment manifest that another engineer could execute.

## Primary sources and canonical code paths

- [PyTorch installation selector](https://pytorch.org/get-started/locally/), [C++/CUDA extension documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html), and [`torch/utils/cpp_extension.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py).
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html), [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/), and [CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- Inspect [`torch/utils/collect_env.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/collect_env.py), [`torch/csrc/utils/cpp_extension.cpp`](https://github.com/pytorch/pytorch/tree/main/torch/csrc/utils), and PyTorch CI workflows under [`.github`](https://github.com/pytorch/pytorch/tree/main/.github).

## OSS bridge and next

PyTorch contributors need reproducible environments before debugging a dispatcher, compiler, or CUDA failure. Practice by running a selected unit test with `pytest`/the project’s documented test command and preserving its build logs. Next, strengthen the Python control plane so it does not dominate an otherwise fast GPU system in [Module 03](03-python-performance-and-data-model.md).
