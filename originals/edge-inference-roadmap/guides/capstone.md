# Capstone: a measured offline document assistant

## Product boundary

Import a small public or synthetic document collection. Retrieve relevant passages locally. Generate a short answer with source references using an on-device model. Show a useful failure when evidence or resources are insufficient. Work in airplane mode after assets are installed.

Begin with text or pre-extracted PDF content. Your existing OCR experience is an advantage, but adding complex page reconstruction too early would obscure the new inference learning. Full OCR, large tables and multimodal reasoning are optional later branches.

## Acceptance levels

**Level 1:** native LLM generation on the phone with manually supplied context. **Level 2:** local index and retrieval with host-precomputed embeddings. **Level 3:** embeddings and generation both on-device. **Level 4:** lifecycle reliability, measured budgets and repeatable evaluation. Clearly state the achieved level rather than presenting host-precomputed work as fully on-device computation.

## Data model

Keep Document records with stable ID, source name, content hash and import version. Keep Chunk records with document ID, source location, text and chunking version. Keep an IndexManifest with embedding model/revision, normalization policy, dimensions, metric and corpus hash. Keep RequestTrace records with retrieved chunk IDs, prompt token count, timings and outcome.

Use a local store such as SQLite for metadata if useful. For a small corpus, an exact vector scan is an understandable and sufficient baseline. Add approximate indexing only after measuring the exact baseline and its quality.

## Request path

Validate request and budgets; embed the query; retrieve candidates; choose source passages under the context budget; construct a versioned prompt; generate with bounded output; validate references or structured output; return answer and evidence. Record per-stage timings and explicit terminal status.

Retrieved text is evidence, not instructions for your application. Keep source content separate from your control prompt and test documents containing instruction-like text. The application's answer policy should preserve source provenance and allow abstention.

## Resource policies to implement

- One generation worker initially, with bounded queue capacity.
- Explicit per-request input and output token limits.
- No model switching while an active session still owns its buffers.
- Conservative embedding/generator residency based on measured peaks.
- Cancellation at known safe points with a single terminal event.
- Recovery after process recreation without assuming native state survives.
- Temporary asset/index writes followed by validation and activation.

## Evaluation set

Create approximately 50 reviewed questions. Suggested composition: 20 direct factual, 10 requiring multiple passages, 10 unanswerable, 5 with similar distracting facts and 5 long-context or formatting-sensitive cases. Adjust to the corpus; the goal is coverage, not a universal benchmark ratio.

For each item store question, answerability, expected facts, gold source locations, acceptable answer variation and scoring notes. Keep a held-out subset. Do not repeatedly tune on it and still describe it as untouched evaluation.

## Quality measures

Retrieval recall at k asks whether required evidence is present in the selected set. Answer correctness checks expected facts. Support rate asks whether claims follow from evidence. Citation validity checks that references exist and point to supporting passages. Abstention scoring distinguishes appropriately declining from failing an answerable question.

Use clear denominators: five supported claims out of six evaluated claims differs from five fully supported answers out of six questions. Report human review and any judge model separately. Preserve disagreement examples.

## Performance measures

Report ingestion/index time separately from query time. Split query embedding, retrieval, prompt assembly, load/preparation, prefill and decode. Record application TTFT and total latency, context size, output size, available memory measurements and failure count. Include a sustained workload rather than only a single cool-device request.

Targets should be selected after your baseline. Do not invent a guaranteed token rate from the chipset specification. A smaller model with better retrieval and predictable latency may be the best deployment even if a larger model fits.

## Release demonstration

From a clean launch: select a valid installed model; import or load the corpus; ask an answerable question; inspect citations; ask an unanswerable question; cancel a long request; restart the application; simulate a missing or corrupt asset. Save the exact build and evaluation metadata.

## Portfolio write-up

Explain the problem, architecture, model/backend decision, memory ownership, benchmark method, quality results and observed failures. Include one concrete before/after optimization with raw evidence. Distinguish your educational engine from the established runtime used to deliver the final app.

The capstone is strong when another developer can reproduce a limited, well-defined result. It does not need to claim universal document understanding or perfect extraction.
