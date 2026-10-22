# Reading and source catalogue

Checked core references and curated supplementary entry points. Dynamic pages can change; pin software versions and keep the matching documentation. Read by topic rather than treating every linked site as required cover-to-cover reading.

## cpp: C++ language curriculum

https://www.learncpp.com/

Use the site index to find the daily topic; chapter numbering can change. Read the relevant subsection, not an entire chapter by default.

## guidelines: C++ Core Guidelines

https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines

Reference for resource management, ownership and concurrency. Read a handful of applicable rules after your own design.

## cmake: CMake tutorial

https://cmake.org/cmake/help/latest/guide/tutorial/index.html

Follow the tutorial matching your installed stable CMake version; latest documentation can describe newer releases.

## debug: Clang sanitizers

https://clang.llvm.org/docs/AddressSanitizer.html

Start with AddressSanitizer; linked guides explain other diagnostics and platform restrictions.

## android: Android NDK

https://developer.android.com/ndk/guides

Primary reference for native builds, ABI, native APIs and tooling.

## jni: Android JNI tips

https://developer.android.com/training/articles/perf-jni

Read ownership, references, threads and exception guidance before integrating native generation.

## llama: llama.cpp Snapdragon guide

https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md

Current upstream accelerator route. Pin a commit and use the instructions in that checkout; backend support remains device-dependent.

## bench: llama.cpp repository and benchmark sources

https://github.com/ggml-org/llama.cpp

Inspect benchmark tools and their help at the pinned commit. Do not copy flags from unrelated versions.

## perf: Android Simpleperf

https://developer.android.com/ndk/guides/simpleperf

Use supported profiling modes and document permission limits; combine with your own stage timings.

## pybind: pybind11 documentation

https://pybind11.readthedocs.io/en/stable/

Focus on functions, NumPy buffers, lifetime and GIL rules; skip advanced binding features initially.

## torch: PyTorch documentation

https://docs.pytorch.org/docs/stable/index.html

Use tensor, inference-mode and export API references matching your installed release.

## model: SmolLM2-135M official model card

https://huggingface.co/HuggingFaceTB/SmolLM2-135M

An educational checkpoint candidate; inspect config and reference implementation before writing architecture-specific code.

## tokenizer: Hugging Face Tokenizers documentation

https://huggingface.co/docs/tokenizers/index

Read the tokenizer pipeline and use official model tokenizer assets; a hand-written educational tokenizer is a separate exercise.

## llama2c: llama2.c

https://github.com/karpathy/llama2.c

Read only after implementing your first forward pass. This is educational C, not a production C++ architecture template.

## quant: ggml source and quantization implementation

https://github.com/ggml-org/llama.cpp/tree/master/ggml

Inspect format and kernel concepts after your scalar implementation; educational INT4 is not automatically a supported GGUF quant.

## arm: Arm Neon intrinsics reference

https://arm-software.github.io/acle/neon_intrinsics/advsimd.html

Look up exact intrinsics and required architecture features; preserve a scalar reference and correct tail handling.

## executorch: ExecuTorch LLM documentation

https://docs.pytorch.org/executorch/stable/llm/working-with-llms.html

Start with one supported example. Exporter, runtime and backend versions must agree.

## qnn: Qualcomm Genie LLM tutorial

https://github.com/qualcomm/ai-hub-apps/blob/release/tutorials/llm_on_genie/README.md

Target-aware vendor path; verify SDK, firmware and model-asset compatibility on your exact device.

## litert: LiteRT-LM

https://github.com/google-ai-edge/LiteRT-LM

Optional comparison after your main runtime and application work; check supported models and packaging.

## rag: SQLite documentation

https://www.sqlite.org/docs.html

Useful local metadata and index-manifest storage; vector search can start as your own exact scan.

## eval: lm-evaluation-harness

https://github.com/EleutherAI/lm-evaluation-harness

Optional reference for reproducible language-model task evaluation; your capstone still needs source-grounded custom tests.

## release: Android application fundamentals

https://developer.android.com/guide/components/fundamentals

Review lifecycle and packaging requirements for your application. This curriculum is not a Play Store publishing checklist.

## Additional primary references

- Android native cross-compilation: https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md
- ExecuTorch Qualcomm backend: https://docs.pytorch.org/executorch/stable/android-qualcomm.html
- Android LLM runner: https://docs.pytorch.org/executorch/stable/llm/run-on-android.html
- Perfetto: https://perfetto.dev/docs/
- UndefinedBehaviorSanitizer: https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
- ThreadSanitizer: https://clang.llvm.org/docs/ThreadSanitizer.html
- C++ API reference: https://en.cppreference.com/w/
- Google Benchmark: https://github.com/google/benchmark
- Catch2: https://github.com/catchorg/Catch2
- Llama 3.2 1B Instruct model card: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Gemma 3 1B IT model card: https://huggingface.co/google/gemma-3-1b-it

These models are stable educational candidates, not a ranking of the newest checkpoints. Architecture, licensing and backend support should determine the final selection. The curriculum's small decoder implementation targets exactly one chosen architecture initially.

## How to read source code

Start at the CLI entry and follow one request. Identify model loading, tokenization, batch construction, graph execution, cache mutation and sampling. Read a single operator implementation with its test. Write a call-path note before reading another subsystem. For a large file, search for the symbols on your call path rather than reading every line sequentially.

For papers, write the problem, assumptions, algorithm, implementation implications and one experiment you could reproduce. Save elaborate paper study for the optional specialization stage if it blocks the core implementation.
