# Module 48 — llama.cpp/ggml Internals and Contributions

**Phase:** 7 — Distributed runtimes and open source  
**Pacing:** Week 24, paired with Module 47; three 3-hour sessions plus the final contributor-readiness gate  
**Prerequisites:** Modules 17–27 and 35–47; model forensics, GGUF/quantization, graphs, CPU/GPU kernels, benchmarking, and source-walk practice

## Why this matters

llama.cpp exposes inference mechanics: GGUF loads into model objects, ggml constructs a compute graph, a scheduler assigns graph splits to CPU/GPU devices, backend kernels execute quantized block operations, and server slots drive parallel decoding. The project explicitly requires human understanding and forbids AI-authored project communication or autonomous submissions.

## Outcomes

You can:

- trace GGUF metadata/tensors into a llama model and ggml graph;
- reason about ggml dimension order, `ggml_mul_mat`, quantized blocks, strides, and backend buffers;
- follow graph allocation, backend support checks, graph partitioning, execution, and sampling;
- benchmark prompt processing and token generation across CPU/GPU offload choices;
- prepare a contribution consistent with current cross-backend tests and AI-use policy.

## Mental model: file → model → graph → backend

GGUF stores typed metadata plus aligned tensor descriptors/data. The loader maps architecture-specific keys and tensor names into model hyperparameters and tensors, often using memory mapping. Quantized types are block formats: a logical row contains groups of codes plus scales/metadata, so storage and kernel indexing use type traits and block sizes rather than “bits times elements” alone.

ggml tensors store up to four dimensions with byte strides. Project convention calls dimension 0 columns, dimension 1 rows, and dimension 2 matrices; dimension 0 is contiguous in a standard row. Matrix multiply notation is intentionally worth deriving. The contribution guide states:

\[
C=\texttt{ggml\_mul\_mat}(A,B)
\quad\Longleftrightarrow\quad
C^\top=AB^\top
\quad\Longleftrightarrow\quad
C=BA^\top.
\]

A model-specific graph builder creates operations—embedding lookup, normalization, matmul, RoPE, attention, FFN, logits—without immediately executing them. A graph allocator assigns buffers. The backend scheduler queries which backend supports each operation/buffer, partitions the graph, inserts copies as required, and invokes backend execution.

CPU, CUDA, Metal, Vulkan, and other backends need not support identical operations or quantized types. Partial GPU offload places selected tensors/layers in backend buffers; crossing a backend boundary costs copies and synchronization. Prompt processing uses larger GEMMs and batches; token generation repeatedly executes small-\(M\) work and KV attention. `llama-bench` reports these regimes separately, so never summarize both as one speed number.

At the server layer, slots represent concurrent sequences around a shared model/context policy. Trace request parsing, slot assignment, tokenization/chat templates, batching/decode, sampler chain, stop handling, and streaming independently from the ggml compute graph.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/gguf_inspect.py` using the project's supported GGUF Python tooling to list metadata, tensor names, logical dimensions, types, offsets, and calculated row/storage sizes. Compare with official CLI inspection; do not write a permissive untrusted-file parser casually.

### 2. Optimized

In a disposable pinned fork, add a graph evaluation callback or use existing diagnostics to emit node op, shapes/strides, assigned backend, buffer type, and copies for one tiny model. Reconstruct a small `ggml_mul_mat` with a dense oracle and one quantized row.

### 3. Integrated

Run `llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`, and `test-backend-ops` as applicable. Sweep CPU threads, batch/ubatch, GPU-offloaded layers, and one supported quantization. Pair every speed result with model hash, quality evidence, selected backend, and memory.

## Three 3-hour sessions

1. **GGUF and model:** inspect metadata/tensors, trace architecture dispatch and loading, and derive graph-node shapes.
2. **Graph and backend:** follow graph construction, allocation, backend scheduler splits, copies, quantized traits, and CUDA/CPU kernels.
3. **Contributor loop:** identify a small backend-op or model edge case, create a failing test/reproducer, run cross-backend/quality/performance checks, and prepare a human-owned proposal.

## Experiments and deliverables

Commit only to the learner lab:

- `reference/gguf_inspect.py`;
- `tests/test_ggml_shape_convention.py`;
- `benchmarks/results/llamacpp-offload-quant.json`;
- `benchmarks/reports/llamacpp-pp-tg.md`;
- `notes/source_walks/gguf-to-backend.md`;
- `notes/contributions/llamacpp-proposal.md`.

Record llama.cpp commit, build options/backends, compiler/toolkit, model/GGUF hash, metadata and quantization type, tensor placement, threads, batch/ubatch/context, GPU offload/split, prompt-processing and generation tokens/s, memory, perplexity/KL where relevant, repetitions, and raw `llama-bench` output.

## Practical contribution ladder

1. Reproduce a bug and add the smallest backend-op or model test.
2. Fix a bounded existing operator/model path across supported platforms.
3. Add model architecture support following the official guide, normally starting with CPU support.
4. Optimize one backend with `test-backend-ops`, quality, and `llama-bench` evidence.
5. Propose a new quantized type/backend only with prior discussion and its much larger portability, conversion, PPL/KL, and performance burden.

Current contribution rules require searching issues/PRs first; features begin with an issue; bug fixes include a reproducer and regression test; new contributors keep one open PR and avoid trivial changes. Run local CI, `llama-perplexity`, `llama-bench`, and `test-backend-ops` when applicable. Prefer simple, portable changes and separate unrelated work.

## Official AI-use rules

Read both current `CONTRIBUTING.md` and `AGENTS.md` before doing contribution work. AI-generated code is permitted only when the human fully understands, manually reviews, can debug, explains, maintains, and discloses it. The policy prohibits AI-written bug reports, feature requests, PR descriptions, commit messages, discussions, and reviewer responses. Automated commits/PR submissions are prohibited and may lead to a ban; autonomous agents are told not to contribute. Therefore an assistant may help the learner study or test locally, but the learner must design the change and personally write all project communication. Do not ask an agent to open or respond to a llama.cpp PR.

## Common failure modes

- Reading ggml dimensions as conventional framework `[rows, columns]`.
- Treating a quantized block type as independent scalar elements.
- Adding a graph op without backend support/fallback and allocation tests.
- Hiding backend copies inside a claimed GPU speedup.
- Comparing prompt-processing tokens/s with token-generation tokens/s.
- Testing only CUDA and breaking CPU/Metal/Vulkan behavior.
- Adding a quantization type without conversion, PPL/KL, storage, and kernel evidence.
- Opening a feature PR before an issue or without long-term ownership.
- Using AI to write project-facing communication or submit automatically.

## Mastery gate

Given a pinned GGUF, trace one tensor from metadata/offset through model placement into a named graph operation and backend kernel. Derive its ggml matrix shapes, verify a dense/quantized oracle, and explain every backend split/copy. Produce reproducible prompt-processing and generation results plus quality checks. Finally, state the exact current contribution and AI-policy constraints and present a proposal entirely in your own words.

## Canonical code areas and OSS bridge

Trace public APIs in `include/llama.h`; model loading and architecture graph builders under `src/`; GGUF/ggml tensors, allocators, backend scheduler, quantization traits, and backend implementations under `ggml/`; conversion in `convert_hf_to_gguf.py` and `gguf-py/`; server slots in `tools/server/`; benchmarks/tests in `tools/llama-bench/`, `tools/perplexity/`, and backend-op tests. Follow symbols and types rather than brittle line numbers.

## Primary and authoritative sources

- [llama.cpp source](https://github.com/ggml-org/llama.cpp)
- [llama.cpp contributing guide](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- [llama.cpp agent and AI-use policy](https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md)
- [Official guide to adding a model](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)
- [`llama-bench` documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

## Next

The 48-module core is complete. Choose one primary runtime, finish the capstone contribution in a human-owned branch, and use the extension roadmap for deeper kernels, distributed serving, multimodality, or production operations.
