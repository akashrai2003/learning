# Interview preparation for the next offer

## Prepare for the role you can credibly win

Use Python as your default interview language unless the role explicitly tests another language. Your immediate bottleneck is likely consistent coding, backend depth, system-design reasoning and defensible project stories, not knowing every inference paper. Continue C++ fundamentals, but do not switch all coding practice to beginner C++ while pursuing Python-heavy jobs.

Confirm each employer's process early. Some use algorithmic interviews; others use practical tasks, debugging or take-homes. The structure below is your preparation plan, not a claim that every company uses these rounds.

## Coding practice method

For a 60-minute session: 5 minutes restate requirements and examples; 10 minutes choose a data structure and complexity; 25 minutes code; 15 minutes test and explain; 5 minutes log the failure category. Revisit missed questions after two days and one week. Count independently solved and explained problems, not merely watched solutions.

Cover arrays/maps/sets, two pointers, sliding windows, sorting, binary search, stacks/queues, heaps, tree traversal and graph BFS/DFS. Add basic dynamic programming when target interviews require it. Aim initially for approximately 30–40 well-reviewed problems over the cycle, supplemented by practical tasks. Increase or redirect according to real interview feedback rather than a large arbitrary problem count.

## Practical Python drills

1. Implement an LRU cache with specified eviction and empty-capacity behavior.
2. Merge streamed text fragments and handle incomplete records without data loss.
3. Implement a token-bucket rate limiter with an injected clock for deterministic tests.
4. Build a bounded async worker pool with cancellation and clean shutdown.
5. Retry a transient operation with bounded attempts, jitter and an overall deadline.
6. Deduplicate idempotent ingestion requests while detecting conflicting payloads.
7. Parse benchmark records and preserve failures and missing values.
8. Implement top-k exact vector search and compare it to a reference.
9. Create a batcher with a maximum size and maximum wait time; explain latency tradeoffs.
10. Design a small model-switch state machine and test cancel/load failure transitions.
11. Write a streaming HTTP client with timeout and error handling against a local mock.
12. Repair an intentionally blocking call inside an async service and explain the alternative.

Implement a narrow version in one session and deepen tests later. Do not pretend a minimal rate limiter is a distributed production implementation. Explain what additional coordination or persistence would be needed.

## Python depth checklist

Understand mutable defaults, references and copying, iterators/generators, context managers, exceptions, typing, dataclasses, dependency boundaries and testing. Explain the GIL without claiming that async makes CPU work parallel. Distinguish concurrency, parallelism, threads and processes. Explain backpressure and cancellation propagation in an async service. Read code as well as writing new functions.

## Backend and data fundamentals

Practice HTTP semantics, status codes, streaming, retries, idempotency, queues, dead-letter handling, timeouts and graceful shutdown. Know how to choose object storage versus a relational/document database and how index/version metadata fits together. Practice SQL joins, grouping, window functions, transaction basics and index reasoning. Use a local small dataset; do not claim database tuning expertise from memorized terms.

Your RabbitMQ/MongoDB/MinIO experience gives you real stories. Explain delivery guarantees, duplicate handling, consistency boundaries and recovery, not just which components were in Docker Compose. Know where an operation can succeed before a client timeout and why retrying can duplicate work.

## LLM fundamentals to explain clearly

- Prefill versus decode and why their workloads differ.
- Weight memory versus KV cache versus activations and workspace.
- GQA and context-length effects on cache size.
- Quantization formats, error and kernel support.
- Continuous batching versus request batching and token-processing limits.
- Prefix caching and exact invalidation conditions.
- Speculative decoding: acceptance and drafter overhead, not guaranteed speedup.
- Tensor, pipeline and expert parallelism at conceptual level; distinguish reading from operating experience.
- Tokenization, chat templates, stop tokens and constrained output.
- Retrieval quality, reranking, grounding, citation validity and abstention.
- Evaluation design, leakage, held-out sets and limits of judge models.
- Fine-tuning versus retrieval versus prompting as responses to different failures.

For each, prepare a plain-language explanation, a small calculation or example, and a failure case. Use your own public experiments when possible.

## Four system-design cases

**Case A — document intelligence:** ingest large documents, store versions, run extraction, retrieve evidence, answer with citations and support review. Discuss incremental ingestion, document provenance, idempotency, retries, evaluation and updates. Separate indexing from query latency.

**Case B — constrained inference service:** serve a small model on limited GPU memory with short and long requests. Discuss admission control, context/output limits, queueing, batching, cache, cancellation, observability, overload and quality. State SLOs as assumptions, estimate capacity, then propose a benchmark.

**Case C — model residency and switching:** share a GPU between OCR and a generator. Define ownership and state transitions, requests in flight, unload/load, warmup, cache validity, failure rollback and readiness. This is close to your experience; explain the invariant that prevents work from touching unloaded state.

**Case D — remote AI product backend:** support multiple tenants, streaming responses, retries, metering and evaluation. Discuss isolation, rate limits, auditability, timeout budgets, API compatibility and rollout. Avoid naming infrastructure without explaining the failure it solves.

For a 45-minute design: 5 clarify, 5 estimate and define SLOs, 10 data flow and APIs, 10 critical tradeoffs, 10 failure/measurement discussion, 5 summarize. Adjust to interviewer steering. Draw a compact diagram and explain what each boundary owns.

## Debugging cases

1. Throughput drops after a runtime upgrade: compare workload/build/backend, graphs, memory and scheduler behavior before changing settings.
2. TTFT rises but decode is unchanged: inspect queueing, preparation, tokenization and prefill, with timing boundaries.
3. OOM occurs only on long requests: inspect cache/workspace growth, concurrency and admission policy.
4. Cancelled requests retain memory: inspect ownership, outstanding tasks/callbacks and allocator behavior.
5. An OCR result looks plausible but has wrong columns: isolate structural correctness from text recognition and use source-grounded evaluation.
6. A quantized model is smaller but slower: verify kernel/backend support, unpacking and data movement.
7. A queue retries indefinitely: classify permanent/transient failures and impose retry/dead-letter policies.
8. A judge score improves while user trust falls: inspect rubric, unsupported claims, dataset leakage and human review.

Answer with hypotheses, discriminating observations and the next smallest experiment. Do not guess a configuration flag as the whole diagnosis.

## Mock scorecard

Score 0–3 each for clarification, correctness, test coverage, complexity/tradeoffs, explanation and recovery from mistakes. Save the actual failing part: missing edge case, weak Python, slow reasoning, vague ownership or unsubstantiated metric. One mock followed by a repaired implementation is more valuable than repeatedly performing the same weak answer.

## Take-home assessments

Clarify expected time, evaluation criteria, allowed tools and deliverables. Prioritize a correct small implementation, tests, clear README and limitations. Do not expand into an unpaid production service without agreed scope. Follow any explicit AI-use rules. Use human-authored explanations you can defend and avoid exposing employer code.

## Remote collaboration evidence

Practice an asynchronous update containing current status, evidence, blocker, options and next action. Write a small RFC that another engineer can review without a call. Explain time-zone availability honestly and ask about required overlap. Good written engineering communication is both an interview skill and a useful public artifact.
