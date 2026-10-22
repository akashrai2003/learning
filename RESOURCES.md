# Resource guide

## Read just enough for today's task

Each daily plan links a local lesson and a direct external entry from that lesson or a matching official topic guide. Start with the local explanation and the specified subsection; the external entry is for the same 20–30 minute reading window. Follow its source section if you need the exact API, algorithm or production implementation. Do not read every source attached to a module before coding.

The first local resource is primary. Additional local links connect the same topic across courses or supply an optional detail. For a source walk, choose one entry point, follow one operation/request, find its test and record the repository commit. For a paper, extract the mechanism and assumptions needed by the day's experiment; a full literature review is not required.

## The source library you already have

| Need | Best existing index |
|---|---|
| Ordered PyTorch explanations, design docs and code paths | [PyTorch source atlas](originals/pytorch-systems-and-deployment-curriculum/resources/SOURCE_ATLAS.md) |
| Framework, compiler, export and distributed references | [PyTorch primary sources](originals/pytorch-systems-and-deployment-curriculum/resources/PRIMARY_SOURCES.md) |
| Transformer, quantization, attention, CUDA and serving papers | [Inference primary sources](originals/inference-engineering-curriculum/resources/PRIMARY_SOURCES.md) |
| C++, Android, native diagnostics and edge backends | [Edge resources](originals/edge-inference-roadmap/resources.md) and [C++ learning map](originals/edge-inference-roadmap/guides/cpp-learning-map.md) |
| Role research, interviewing and contribution context | [Career sources](originals/ai-career-acceleration/sources.md) and [research snapshot](originals/ai-career-acceleration/guides/research-snapshot.md) |

These indexes remain references, not five new reading queues.

## Core external entry points inspected while assembling this plan

Checked on **22 October 2026** using web retrieval. “Inspected” means the page contents were accessible, not that a tutorial was executed or that compatibility with your machine was verified.

| Topic | Entry | Use |
|---|---|---|
| PyTorch foundations | [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Fill specific tensor/model/training gaps |
| C++ tutorial | [LearnCpp](https://www.learncpp.com/) | Find the named daily topic in its index; a teaching resource, not a language specification |
| C++ ownership and interfaces | [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) | Consult the relevant resource-management/interface rules |
| Builds | [CMake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) | Build a small target/library project; use docs matching the installed version |
| Quantization | [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html) | Supported workflows, configuration and tensor representation |
| Triton | [Official tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | Vector addition, reductions and matmul examples |
| CUDA | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | Execution, memory and synchronization references |
| Native custom ops | [PyTorch C++/CUDA operators tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) | Registration, implementation and correctness integration |
| Serving | [vLLM documentation](https://docs.vllm.ai/en/latest/) | User/developer/benchmark references for the primary runtime |
| Exported execution | [ONNX Runtime documentation](https://onnxruntime.ai/docs/) | Artifact execution, providers and measurement guidance |
| Edge runtime | [ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/index.html) | Export, lowering, runtime and backend entry points |
| Android native deployment | [NDK guide](https://developer.android.com/ndk/guides) | Native build and target integration concepts |

The stable [compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html), [export](https://docs.pytorch.org/docs/stable/export.html), [tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html) and [autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html) URLs responded with redirects in the research tool. Compiler/export destination retrieval did not provide readable content in that tool. They are useful inherited documentation pointers, but this plan does **not** treat their full current content as freshly verified. Open the version-matched docs in your normal browser when those sessions begin.

## Verification boundaries

The many individual paper URLs, repository paths, model cards and additional daily external entries inherited from your source packs were **not all rechecked**. The local links in the new roadmap were checked automatically. Original files, including any pre-existing stale external or internal links, were kept untouched. A source title or URL in an original pack is not confirmation that a particular model, API, backend or feature remains available today.

Before an implementation, record the version/commit you use and follow that checkout's documentation. Before a contribution or application, recheck the current contribution policy or official job posting. The new plan does not revalidate old salary/hiring assertions, reserve issues, require a particular future model or recommend a paid service.

## Avoid resource overload

Use one lesson, one implementation and one independent check per session. A longer paper or alternative runtime becomes optional if it prevents completing the core task. Use [EXTENSIONS.md](EXTENSIONS.md) for deferred depth and [COVERAGE.md](COVERAGE.md) to see why a module is a survey rather than a full assignment.
