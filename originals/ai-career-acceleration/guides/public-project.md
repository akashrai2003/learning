# Flagship project: reproducible inference operations lab

## Why this project

Your phone curriculum is a longer learning path. For the next offer, build a smaller public artifact that combines current serving experience with software-engineering evidence. Suggested project: **a benchmark and reliability lab for OpenAI-compatible local inference**, with an optional Android target later.

The question: how do context, concurrency, model residency and request handling affect latency, throughput, failures and memory under constrained hardware? Do not build a new inference engine just to answer it. Reuse an established runtime and add a well-tested client, experiment specification and report.

## Minimum scope for four weeks

Week 1: one runtime, one public model, one fixed prompt fixture and raw-record capture. Week 2: configuration-driven experiments, explicit timing definitions and failure handling. Week 3: a small controlled sweep and parser tests. Week 4: a reproducible report and a 60-second demonstration. This is an estimated development budget; narrow scope if setup blocks you.

## Useful repository structure

Create `src`, `tests`, `configs`, `fixtures`, `reports` and `docs`. The README should explain the question, measured setup, quick start, expected output, limitations and how to regenerate a small report. Keep weights, credentials and private documents out of the repository. A tiny synthetic raw-log fixture is useful for tests.

## Required behaviors

Record requested and actual input/output counts; runtime/model/build identity; request start, first output and completion; timeout/cancel/error status; and available memory measurements with definitions. Preserve raw data before deriving summary metrics. A failed request must remain a failed request rather than a zero or omitted observation.

Use subprocess argument arrays or a structured HTTP client. Validate configuration. Set bounded concurrency and an overall run budget. Support stopping a run and preserving completed records. A complete distributed load generator is beyond the first scope.

## Tests that matter

Mock a streaming endpoint with successful fragments, malformed events, delayed first token, midstream failure and cancellation. Check that timing and failure records match the contract. Test parser behavior independently of expensive model execution. For a real runtime, run a tiny smoke experiment and record version/hardware explicitly.

## Two strong case studies

**Case 1:** loaded/resident versus fresh-process request paths. Separate model preparation and filesystem-cache assumptions. **Case 2:** short versus long context with controlled generation and concurrency. Explain where latency changes and why a single throughput number hides user experience.

An optional third case investigates model switching, provided the runtime supports it and you can validate state safely. Do not imply a bug in an upstream runtime from a configuration mistake; reduce and reproduce first.

## Evidence and content reuse

One experiment can produce a repository test, a report, a LinkedIn post and an interview story. These are views of the same work, not four separate projects. Share one honest result including limitations. A failed optimization can become a useful diagnostic article if the method is sound.

## Connection to upstream

If you discover a real accounting, validation or documentation issue, reduce it to a contribution that fits the upstream project. Keep your experiment suite separate unless maintainers want it. Do not open a PR whose main purpose is linking your personal repository. Follow project policies for human-authored text and AI disclosure.

## Optional second artifact: document correctness fixture pack

Use a few synthetic tables with merged headers, split rows and repeated labels. Specify expected structure and source positions. Compare extraction/export behavior with an independently written evaluator. This supports document AI roles and Docling contribution work. It is optional; one completed flagship is better than two unfinished ones.

## Definition of done

A new reader can identify the exact question, run a small supported experiment, inspect raw records, reproduce one report, understand failures and see the limits. Stars and downloads are not acceptance criteria. Do not put 'production ready' on a prototype without the engineering evidence that phrase implies.
