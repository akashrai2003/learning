# Production engineering and debugging playbook

## Ownership design before performance

Use an immutable Model for weights and configuration, a Session for cache and position, a Workspace for temporary buffers, and a Request for input/output limits and cancellation. This is a useful educational split, not a mandatory class hierarchy.

Models may be shared if the runtime supports that safely; mutable sessions should not be shared accidentally. Borrowed views must not outlive weights, mappings or workspace owners. Document whether callbacks execute synchronously, on a worker, or on a UI dispatcher.

## Error taxonomy

Distinguish invalid request, invalid asset, unsupported configuration, resource limit, cancelled operation, backend initialization failure and internal failure. Preserve diagnostic detail for developers while showing actionable short messages to users. Do not convert every error into an empty successful answer.

At ABI boundaries, contain C++ exceptions. Clean resources with RAII. Avoid destructors that throw. Use a deliberate result protocol where the interface requires one. A catch-all is a final containment boundary, not the normal implementation of error handling.

## Debugging ladder

1. Reproduce with the smallest input and stable versions.
2. Classify compile, link, loader, runtime, numeric or lifecycle failure.
3. Remove accelerator and concurrency variables if possible.
4. Find the first wrong state or tensor, not the last visible symptom.
5. Fix the root cause and add a regression case.
6. Restore one variable at a time and remeasure.

### Compiler error

Read the first relevant diagnostic, locate the smallest failing expression and verify types. Do not repeatedly rewrite surrounding code. A template error can originate at your call site even if the longest message comes from a library header.

### Undefined symbol or library load failure

Check target ABI, link inputs, exported symbols, runtime library paths and dependency versions. A host library cannot simply be copied into an Android ARM64 process. Explain the distinction between a successful build and a successful target load.

### Segmentation fault

Use symbols and a debugger or sanitizer on a reproducible host case. Inspect bounds, lifetime, invalidated views and alignment. Increasing memory allocation randomly is not a diagnosis.

### Incorrect tokens

Compare token IDs and special tokens first, then architecture/config, embeddings, first layer, final logits and cache position. Disable sampling and use fixed input IDs. A different chat template can change output without a numerical kernel bug.

### Slow decode

Check release build, actual backend, thread settings, allocations, context length, fallback, thermal conditions and synchronization. Profile before changing kernels. A larger model or quant that fits can still be slower because of data movement or unsupported compute.

### Freeze on stop or shutdown

Check lock order, waits without predicates, callbacks invoked under locks and workers waiting on a queue that never receives a shutdown notification. Define state transitions and terminal events before adding timeouts everywhere.

### Growing memory

Separate intentional model/cache residency from leaked ownership. Inspect repeated load/unload and request cycles, outstanding callbacks, retained JNI references and queues. Allocators may retain freed blocks, so stable high RSS is not automatically a leak. Look for unbounded growth and ownership evidence.

## Release checks

Keep host numerical tests and boundary tests fast. Add targeted Android lifecycle tests for behavior that cannot be established on host. Save model/runtime compatibility metadata and validate assets before activation. Download into a temporary area, verify, then activate atomically where the filesystem/API permits it.

Separate correctness, performance and quality regressions. Performance thresholds should account for measurement variance and device state; avoid a brittle 'must be faster every run' check. Quality evaluation should retain examples that previously failed.

## What not to overbuild

You do not need Kubernetes, distributed tracing infrastructure, microservices or a network server for a phone-only learning app. A local structured log, simple benchmark runner, small native API and explicit state machine can expose the relevant systems engineering clearly.

Production quality comes from maintained contracts and evidence, not adding infrastructure names. Your existing deployment experience is useful when deciding how little machinery the target actually needs.
