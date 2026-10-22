# Implementation lab contracts

These are specifications, not completed solutions. Daily sessions build them incrementally. Keep reference versions when optimizing.

## Lab A: Device benchmark runner

**Purpose:** turn an ad hoc phone experiment into a reproducible measurement.

Inputs: device selector, runtime path, model identifier/hash, backend, prompt fixture, generation settings, repetition count and timeout. Use subprocess argument arrays. The runner must not interpret prompt text as shell code.

Outputs: raw stdout/stderr, exact arguments, elapsed time, return code, run identity and parsed metrics when available. Separate collection from parsing so parser bugs do not destroy evidence. If a process times out, preserve the partial log and label the run incomplete.

Minimum tests: no device; multiple devices without explicit selection; missing executable; model load failure; malformed runtime output; timeout; successful output; interrupted experiment resume. Parser tests can use small saved log fixtures without a phone.

Acceptance: replay one run from its metadata and regenerate the summary from raw records. Do not impute failed runs as zero latency or quietly remove them.

## Lab B: Tensor owner and view

**Purpose:** learn shape, storage and lifetime contracts.

The owner stores contiguous FP32 values and validated dimensions. The view borrows contiguous storage with compatible shape. Start with row-major layout and a small supported rank. Define zero-sized dimension behavior explicitly.

Operations: element count, checked indexing, reshape preserving element count, dot product and matrix-vector multiplication. Validate multiplication of dimensions before allocation. A reshape changes interpretation, not the underlying count.

Minimum tests: one-dimensional, rectangular and zero-sized cases; invalid index; mismatched dot product; dimension multiplication overflow; view invalidation demonstration in an isolated negative test. Normal application tests must not intentionally execute undefined behavior.

Acceptance: describe exactly who owns storage and whether an API can return a view to a local owner. Reject unsupported striding instead of pretending every input is contiguous.

## Lab C: Checkpoint format and loader

**Purpose:** bridge Python reference data and native code without trusting binary input.

Write a short format specification with magic, version, byte order, tensor count, names, dtype, dimensions and byte lengths. Either store offsets explicitly or define an unambiguous sequential layout. Encode integer fields at fixed widths. Do not dump a host C++ struct and assume it is portable.

Validate the file's total length, every field length, tensor size arithmetic, offset ranges and duplicate names. Bound metadata sizes before allocating. Define alignment requirements for mapped access; copying to properly aligned storage is a valid first implementation.

Minimum tests: bad magic/version; truncated header; truncated data; impossible rank; enormous dimension; duplicate tensor name; unsupported dtype; invalid offset; overlapping ranges if prohibited. A Python exporter creates valid fixtures independently of the C++ parser.

Acceptance: rejected inputs fail with a category and useful detail. The loader cannot return a tensor view whose mapping has already been destroyed.

## Lab D: Small decoder-only engine

**Purpose:** understand the full inference path, not reproduce all features of a mature runtime.

Support exactly one architecture/configuration first. Define embedding, norm, attention, RoPE, MLP, residual and output-head behavior from that architecture. Use fixed token IDs for initial parity. Keep model weights immutable and session cache/workspace mutable.

Minimum tests: each operation, one block, full short prefix, cached versus uncached logits, multiple cache positions, reset, invalid token, full context, EOS, cancellation and independent sessions. Save intermediate tensors for the first divergence rather than judging only generated prose.

Acceptance: model revision and format are recorded; numerical agreement is demonstrated with a justified tolerance; native generation works on host and later Android. General model support is explicitly out of scope.

## Lab E: Quantized matvec

**Purpose:** understand representation, dequantization cost and quality loss.

Implement symmetric INT8 and grouped packed INT4 using a specified signed range, rounding rule, scales, packing order and tail policy. An all-zero group needs a safe scale rule. Keep a scalar dequantized oracle separate from the optimized path.

Minimum tests: every 4-bit code; both nibble positions; odd lengths; tail groups; zeros; clipped outliers; non-square matrices; scale indexing; large accumulations. Measure scales and metadata in storage totals.

Acceptance: errors versus original float weights are separated from errors versus the dequantized reference. Report unpacking time, workspace and overall model performance rather than claiming speedup from smaller weights alone.

## Lab F: Bounded executor

**Purpose:** learn reliable inference infrastructure with limited concurrency.

States: queued, running, cancellation requested, and one terminal result such as completed/cancelled/failed. Decide whether cancelling a queued item removes it immediately or marks it for skipping. Define shutdown as accepting no new work, waking waiters, and finishing or cancelling existing work according to one documented policy.

Minimum tests: queue full; worker waiting; producer waiting; cancel before start; cancel while running; callback throws or fails; shutdown with pending work; repeated shutdown; rejected new work. Callbacks must not run while holding a lock if they can re-enter your executor.

Acceptance: each accepted request has exactly one terminal event and no callback outlives its receiver. ThreadSanitizer is useful on supported host builds but does not prove all possible schedules are safe.

## Lab G: Native Android shell

**Purpose:** host inference without freezing or corrupting the application.

Keep a small JNI interface and document handle ownership, permitted call order, callback thread and cancellation behavior. Do not pass raw pointers as unvalidated public handles without a lifetime strategy. Keep model sessions away from the UI thread.

Minimum tests: repeated start/stop; invalid handle; load failure; Unicode output; screen recreation; backgrounding; model switching; process restart. Preserve valid text when tokens contain partial UTF-8 sequences.

Acceptance: terminal events are coherent, state is recoverable, and diagnostics are useful. Android lifecycle policy is explicit rather than relying on a lucky activity lifetime.

## Lab H: Backend deployment investigation

**Purpose:** prove what runs, where, and with what limits.

Use an official supported example before a custom model. Record model/config/SDK/runtime/device versions. Validate outputs and inspect delegation or runtime logs. A successfully loaded accelerator library is not proof that the main graph executes there.

Minimum evidence: initialization log, executed model, input shape, target backend, quantization, output check, preparation cost and steady-state measurement. If blocked, provide a minimal reproduction and CPU fallback.

Acceptance: classify support as verified, documented but untested, or failed on this configuration. A careful incompatibility report is a valid result.
