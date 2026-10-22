# Module 39 — LibTorch and the C++ Frontend

**Phase:** 6 — Native PyTorch and GPU integration  
**Time:** 12 focused hours  
**Prerequisites:** C++17/CMake, Modules 36–38, and a tested PyTorch model or operator.

## Existing inference curriculum bridge

Relate this to [01 — Inference system map](../../inference-engineering-curriculum/modules/01-inference-system-map.md), [02 — Reproducible stack/toolchain](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), [41 — Custom operators](../../inference-engineering-curriculum/modules/41-pytorch-custom-operators.md), [43 — Build/JIT/Ninja/ABI](../../inference-engineering-curriculum/modules/43-build-jit-ninja-cmake-and-abi.md), [45 — Distributed inference](../../inference-engineering-curriculum/modules/45-distributed-inference-parallelism.md), and [48 — llama.cpp/GGML](../../inference-engineering-curriculum/modules/48-llamacpp-ggml-internals-and-contributions.md).

## Why this matters and outcomes

LibTorch is the C++ distribution/front-end for PyTorch: tensors, modules, autograd, optimization, data, serialization, and custom operators are available without a Python application loop. It is useful for low-latency embedding, native services, integration tests, and learning framework internals. It is not automatically faster than Python—most Python tensor work already calls C++—so choose it for integration/control, then measure.

You will build a small CMake application against a pinned LibTorch distribution, move a tested module/operator boundary to C++, establish model/artifact parity, use inference-mode semantics correctly, and diagnose packaging/ABI problems before treating C++ as a deployment panacea.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P39 — LibTorch and C++ frontend**.

1. [PyTorch C++ frontend tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html) — learn the API or system contract.
2. [C++ API docs](https://docs.pytorch.org/cppdocs/) — study the underlying design, paper, or specification.
3. [TorchScript deprecation note](https://docs.pytorch.org/docs/stable/jit.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch custom-ops landing page](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch extension-cpp examples](https://github.com/pytorch/extension-cpp) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch dispatcher walkthrough](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

The C++ frontend includes `torch::Tensor`, `torch::nn`, `torch::optim`, `torch::data`, and serialization APIs, built over ATen/c10. A C++ `Module` registers parameters and buffers much like Python. The essential distinction is semantic ownership: loading a raw `state_dict` only restores values into a matching architecture; it does not reconstruct arbitrary Python model code. A robust deployment artifact must state architecture/version, parameter names/shapes/dtypes, tokenizer/processor/preprocessing, custom op libraries, and target runtime/library versions.

For inference, disable gradient recording intentionally (`NoGradGuard`/inference-mode APIs appropriate to the pinned frontend) and call `eval()` so dropout/normalization behavior matches the source model. `eval()` does not disable autograd, and no-grad does not change training-mode modules—use both as needed. Tensor shapes, device placement, dtype, memory format, stream context, and output postprocessing must still match Python. Copying data between CPU and GPU or converting a `std::vector` per request can dominate a supposedly native path.

CMake is the usual integration route: a LibTorch distribution supplies headers, libraries, and CMake package metadata, and `find_package(Torch REQUIRED)` resolves them when `CMAKE_PREFIX_PATH` points to an absolute, matching distribution path. Link your custom-op shared object so its static registrations are retained/loaded; a successful link does not prove registration happened. Pin compiler, C++ standard library ABI, LibTorch build, CUDA runtime, GPU architecture, and dynamic-loader search path. Module 40 is the forensic guide for those boundaries.

For new portable edge workflows, evaluate `torch.export`/ExecuTorch and ONNX/TensorRT paths in their dedicated modules rather than assuming a Python checkpoint plus C++ frontend is a universal interchange format. TorchScript/JIT artifacts exist in ecosystems, but capability and lifecycle must be verified against the exact target release; do not choose a serialization path based on a stale tutorial.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/libtorch/python_oracle.py` with a tiny MLP or vision block, explicit preprocessing, fixed test fixtures, checkpoint manifest, and outputs saved as a portable numeric fixture.
2. **Optimized:** create `torch-lab/libtorch/app/` with CMake, a C++ frontend implementation or matching operator invocation, `eval()` plus inference/no-grad policy, and tests against the Python fixtures on CPU first, then CUDA.
3. **Integrated:** add the native custom op from Module 36, tokenize/preprocess through a deliberately versioned boundary, and implement a simple request loop with admission, warmup, timings, and graceful capability failure. Compare Python and C++ end-to-end stages instead of merely timing a single matmul.

## Three focused sessions

1. **Build and parity:** configure a clean CMake build with an absolute LibTorch prefix and reproduce Python tensor outputs/parameters on CPU.
2. **Runtime semantics:** add CUDA/inference mode/eval, test CPU↔GPU transfers and memory layouts, and find a mismatch with saved intermediates rather than output-only guesses.
3. **Packaging:** bundle a manifest and custom-op dependency, test in a clean shell/container-like environment, inspect dynamic dependencies, and document target limitations.

## Experiments and deliverables

Commit `app/CMakeLists.txt`, `app/main.cpp`, `artifact_manifest.json`, Python/C++ golden fixtures, `test_parity.py`, `test_clean_runtime.sh`, and `latency_breakdown.csv`. Include source revisions, compiler/CMake/LibTorch/CUDA/driver details, C++ ABI setting, model/processor hashes, dtype/device/layout, model-load/cold/warm timings, host/GPU memory, and dynamic dependencies.

## Common failure modes

- Expecting Python `.pt`/pickle-like artifacts to instantiate an arbitrary C++ architecture automatically.
- Calling `eval()` but retaining gradient tracking, or no-grad while leaving dropout in train mode.
- Linking a custom-op library that the dynamic loader drops or cannot register.
- Mixing a LibTorch distribution and extension built for different PyTorch/CUDA/libstdc++ ABIs.
- Claiming C++ wins while excluding Python’s already-native model computation or data movement.

## Mastery gate

Build a clean C++ executable that matches a Python oracle on fixed fixtures, runs a registered custom op, declares all artifacts/dependencies, and measures load, preprocess, transfer, warm inference, and postprocess. Diagnose one intentional ABI/load failure using evidence rather than environment-variable guessing.

## Primary sources, code paths, and OSS bridge

- [C++ frontend documentation](https://docs.pytorch.org/cppdocs/frontend)
- [C++ frontend tutorial and CMake setup](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html)
- [C++ API reference](https://docs.pytorch.org/cppdocs/)
- [PyTorch C++ extension documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/)

Read `torch/csrc/api/`, `aten/`, `c10/`, C++ examples, and your installed LibTorch `share/cmake/Torch/` files. Useful contribution paths are C++ parity tests, better error diagnostics, minimal packaging/ABI reproductions, and docs. Do not land a C++ rewrite merely because it feels more “native”; show an end-to-end systems reason.

## Next

Module 40 traces the JIT/Ninja/CMake/dynamic-loader chain that makes native artifacts trustworthy.
