# Repository map: where your experience is useful

The suggested tasks below are **contribution ideas**, not confirmed bugs or maintainer-approved scope. Reproduce, search duplicates, read current policies and agree on features before substantial work. Hardware needs are approximate and task-dependent.

## Recommended starting choice

Pick **one primary**: vLLM if your serving incidents produce a small public reproduction; GuideLLM if you enjoy measurement and can isolate client/report logic; Docling if you can reproduce a document-structure/export issue with synthetic inputs. Use your strengths immediately rather than beginning with a difficult kernel simply for repository prestige.

| Repository | Fit and starting surface | Hardware and difficulty | Best timing |
|---|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Serving validation, parser behavior, configuration tests, loading diagnostics and reproducible deployment issues | Some logic CPU-testable; serving/GPU claims need supported GPU validation; medium to high | Primary candidate now |
| [vllm-project/guidellm](https://github.com/vllm-project/guidellm) | Benchmark input validation, failed-request accounting, output records and reporting tests | Many client/report tasks can use mocks; end-to-end benchmarking needs an endpoint; medium | Primary or backup now |
| [docling-project/docling](https://github.com/docling-project/docling) | Synthetic table fixtures, export/provenance correctness and parsing regressions | CPU-friendly for many fixtures; some pipelines need models/GPU; medium | Strong match now |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | Sampling/config tests, API behavior, cache-unit tests and benchmark clarity | CPU unit subsets exist; runtime paths can need GPUs; medium/high | Alternate primary |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Android reproducibility, server behavior, conversion tests and eventual backend ops | CPU available; ARM phone useful; native changes require C++ depth; medium/high | After foundations, or a narrow independently found issue |
| [pytorch/executorch](https://github.com/pytorch/executorch) | Export reproducibility, Android examples, shape errors and target validation | Host plus Android for relevant paths; higher setup cost | Later edge specialization |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Tokenization/template fixtures, model tests, generation corner cases and documented behavior | Some tiny-model tests CPU-friendly; scope/review bar can be high | Selective adjacent lane |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Correctness cases, benchmark reproducibility and later kernels | GPU/architecture-specific validation; higher native/CUDA bar | Later, not first default |
| [ray-project/ray](https://github.com/ray-project/ray) | Serving examples, failure handling and distributed workflow tests | Local tasks possible; distributed claims need appropriate setup | Adjacent infrastructure watchlist |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | State/checkpoint behavior, retry/cancellation examples and targeted regression tests | Many logic tasks CPU-friendly; inspect current policy in repo | Secondary if a genuine workflow issue arises |

## vLLM: convert operations experience into source-level evidence

Candidate investigations: a confusing invalid configuration should fail clearly; a parser should handle a documented boundary; a load/wake behavior should have a regression check; a benchmark should distinguish failed and successful output. These are hypotheses derived from your interests, not statements that upstream is broken.

Start by tracing configuration to validation to the affected component. For a memory/residency issue use a public model and a minimal script. Do not upload private deployment files. Compare your observed runtime version with current upstream before assuming the behavior remains unfixed. Follow the current contributor guide for tests, DCO and AI policy [O1].

## GuideLLM: build a reputation for honest measurement

Its repository describes an LLM deployment evaluation focus [O4]. That makes it a plausible match for your benchmark harness work. Inspect its own architecture and current contribution instructions before proposing another metric or CLI option. Prefer correcting a reproducible accounting or validation problem over inventing a new report format without demand.

## Docling: use synthetic complexity

Construct a tiny document with a merged header, split row or source-location edge case. Establish expected text/structure, compare the public pipeline, then isolate whether the issue is OCR, table reconstruction or export. A deterministic fixture can be more useful than a screenshot of a large private document. Read its contributor guide [O5].

## SGLang: scope matters

Read the current test guide and choose a unit-testable path if you lack the required accelerator. Sampling, metadata and server behavior can offer approachable surfaces, but do not assume they are easy. Source-level caching and backend refactors need behavioral and numeric validation. See [O2].

## llama.cpp: earned native depth

Your phone gives you access to a real Android target. A reproducible build/deployment issue can be useful if it is current and within scope. Backend operations are a later step after scalar reference work and the project's tests. The inspected contribution policy has strict AI-text rules; write your actual GitHub communication yourself [O3].

## Snapshot issue leads: examples, not assignments

The searches on 22 October 2026 surfaced the following. Recheck issue state, comments, linked PRs, assignees and remaining work before acting. Search results did not provide complete reliable status metadata, so none is certified available here.

- [llama.cpp #14909](https://github.com/ggml-org/llama.cpp/issues/14909): an umbrella for missing backend operations. It illustrates an operator-test contribution path; an individual operation may already be implemented. Not the first default task for a C++ beginner.
- [vLLM #39479](https://github.com/vllm-project/vllm/issues/39479): config hashing refactor follow-ups. Potentially Python-facing, but requires understanding cache/config semantics and which subtasks remain.
- [SGLang #28808](https://github.com/sgl-project/sglang/issues/28808): attention backend inheritance refactor. The apparent locality does not remove the need for backend validation; likely a stretch without suitable hardware and architectural context.

Do not copy these issue descriptions into a new issue. The better use is learning how maintainers describe scope and testing, then finding a current achievable task.

## Sources and policy entry points

[O1] https://docs.vllm.ai/en/latest/contributing/

[O2] https://docs.sglang.io/docs/developer_guide/contribution_guide

[O3] https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md

[O4] https://github.com/vllm-project/guidellm

[O5] https://github.com/docling-project/docling/blob/main/CONTRIBUTING.md

[O6] https://github.com/pytorch/executorch/blob/main/CONTRIBUTING.md

[O7] https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md

[O8] https://github.com/flashinfer-ai/flashinfer/blob/main/CONTRIBUTING.md
