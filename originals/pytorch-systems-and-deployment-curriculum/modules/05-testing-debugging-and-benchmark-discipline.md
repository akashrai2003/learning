# Module 05 — Testing, Debugging, and Benchmark Discipline

> **Phase:** I — Engineering foundations  
> **Time:** Week 3; three focused 3-hour sessions  
> **Prerequisites:** Modules 01–04 and basic pytest use

## Existing inference curriculum bridge

This operational discipline extends [Module 05 — `performance-modeling-and-roofline.md`](../../inference-engineering-curriculum/modules/05-performance-modeling-and-roofline.md), [Module 19 — `calibration-error-metrics-and-evaluation.md`](../../inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md), [Module 36 — `cuda-execution-memory-and-streams.md`](../../inference-engineering-curriculum/modules/36-cuda-execution-memory-and-streams.md), and [Module 44 — `profiling-benchmarking-and-correctness.md`](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). You will use its evidence standard for ordinary PyTorch changes long before writing a kernel.

## Why this matters and outcomes

Deep-learning code fails on more dimensions than ordinary business logic: values, shapes, dtypes, devices, aliasing, gradients, serialization, compilation, nondeterminism, and performance can each be correct or broken independently. A one-line “works on my GPU” assertion catches almost none of them. Design tests around contracts and measurement around a claim.

You will be able to construct a testing pyramid, make a minimal reproduction, use PyTorch’s numerical assertions and grad checks appropriately, debug asynchronous CUDA errors without permanently serializing execution, time GPU work with the right clock, and report a statistically defensible performance comparison.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P05 — Testing, debugging, and benchmark discipline**.

1. [PyTorch testing overview](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#testing) — learn the API or system contract.
2. [pytest documentation](https://docs.pytest.org/) — study the underlying design, paper, or specification.
3. [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [Python data model](https://docs.python.org/3/reference/datamodel.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Python profiling](https://docs.python.org/3/library/profile.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: evidence descends and ascends

When a model result is wrong or slow, descend from symptoms to a small operator, then return upward with regression tests:

```text
service/model symptom -> module -> operator -> kernel/runtime
                         fix + oracle
kernel/runtime -> operator parity -> module quality -> end-to-end regression
```

Tests answer “does this contract still hold?” Benchmarks answer “under this workload, did a metric change?” A speedup returning wrong logits is a regression; a cold JIT compile is not throughput.

## Test theory and numerical contracts

Build layers deliberately:

1. **Unit tests** exercise pure behavior: argument validation, shape/dtype/device contracts, and state transitions.
2. **Oracle/differential tests** compare a fast implementation to a small, independent reference across randomized cases.
3. **Property tests** assert invariants such as `softmax.sum(-1)==1` for finite unmasked rows, cache decode equals full forward, or a transpose round trip restores values.
4. **Integration tests** load an artifact, execute a tiny train/eval/decode path, and validate observable behavior.
5. **Regression/performance tests** preserve a bug fixture or detect a large performance change under a pinned protocol.

For low precision, write the acceptance predicate first. `torch.testing.assert_close(actual, expected, rtol, atol, equal_nan=...)` should follow error analysis, not a copied default. Exact checks fit integer IDs, indexing, bit packing, and deterministic CPU logic. Floating checks need maximum error, fraction exceeding tolerance, location, NaN/Inf masks, and input shape/dtype/seed. Gradient checks approximate derivatives through finite differences; they require double precision, small perturbations, deterministic functions, and a smooth-enough operation. They do not validate a training run by themselves.

CUDA execution is asynchronous. A device-side illegal access can appear at a later innocent API call. During diagnosis, set `CUDA_LAUNCH_BLOCKING=1` for a separate reproduction or insert narrow synchronization boundaries; remove them from performance measurements. Use `torch.cuda.synchronize()` only to define a host/device boundary. For kernel timing, record CUDA events in the relevant stream, warm the code path, repeat samples, synchronize after the end event, and report the distribution. `time.perf_counter()` around a launch measures enqueue overhead unless the boundary synchronizes.

A comparison needs fixed model/artifact revision, seed policy, environment, input distributions/shapes, warm-up policy, repetitions, statistic, and error gate. Treat compilation, autotuning, disk loading, and CUDA graph capture as separate cold-start numbers. Preserve raw samples; a 2% change without variance is a hypothesis, not a conclusion.

## Implementation ladder in the proposed `torch-lab/`

1. **Reference:** establish `tests/unit`, `tests/numerics`, and `tests/integration`; add deterministic fixtures and a `assert_tensor_contract` helper that reports shape/dtype/device/strides rather than a generic failure.
2. **Optimized:** write a `benchmarks/harness.py` with CPU and CUDA-event timing modes, warm-up, raw JSON output, and a test ensuring it rejects invalid protocols such as mixed shapes or implicit syncs in the measured loop.
3. **Integrated:** add a `scripts/repro.py` format that turns a saved test fixture and environment manifest into a one-command bug report. Add CI tiers: fast CPU checks per change, CUDA/long tests separately, and optional performance comparison against a stored baseline.

## Three focused sessions

1. **Session A — test architecture (3h):** turn the reference math module’s hand tests into parameterized random tests over sizes, dtypes, and devices. Seed every generated counterexample and save one regression fixture.
2. **Session B — debugging drill (3h):** introduce a wrong attention mask and an in-place aliasing bug. Use assertions, `torch.autograd.set_detect_anomaly` only when relevant, CPU fallback, and a minimal script to localize each. Do not leave anomaly mode enabled in normal execution.
3. **Session C — timing drill (3h):** compare CPU wall-clock launch timing with CUDA event timing for a GEMM; separately measure first call and warm calls. Produce raw samples and explain the difference with the device dependency graph.

## Experiments and deliverables

Create a test matrix for `[CPU, CUDA-if-present] × [fp32, bf16/fp16-if-supported] × [contiguous, transposed] × [small, odd-tail]`. It need not run every Cartesian product in every CI job; document which tier covers each risk. Produce a benchmark report that includes the command, manifest, raw samples, median/p50/p99, correctness tolerance, and a roofline hypothesis. Test that a benchmark fails rather than silently comparing different model variants.

## Common failure modes

- Using `assert torch.allclose(...)` without printing enough context to reproduce a seed and location.
- Testing only square, contiguous, batch-one inputs, which hides almost every indexing/layout bug.
- Calling `empty_cache()` or synchronizing in the hot loop, thereby measuring a different system.
- Treating `CUDA_LAUNCH_BLOCKING=1` as a fix rather than a diagnostic that changes scheduling.
- Using a benchmark’s fastest single iteration, a changing random workload, or cold compilation as a steady-state throughput claim.

## Mastery gate

Take a deliberately faulty operator and deliver: a ≤50-line reproducer, a failing regression test, a diagnosis that names the violated contract, a corrected implementation, and a before/after benchmark that proves the fix did not regress the claimed workload. Be able to explain why the timing method is valid for both CPU and CUDA.

## Primary sources and canonical code paths

- [PyTorch testing utilities](https://docs.pytorch.org/docs/stable/testing.html), [gradcheck mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html), [numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html), and [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html), [`torch.profiler`](https://docs.pytorch.org/docs/stable/profiler.html), and NVIDIA’s [CUDA events reference](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).
- Inspect [`torch/testing/_internal/common_utils.py`](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py), [`test/test_torch.py`](https://github.com/pytorch/pytorch/blob/main/test/test_torch.py), and [`torch/utils/benchmark`](https://github.com/pytorch/pytorch/tree/main/torch/utils/benchmark).

## OSS bridge and next

Upstream maintainers value a failing test and minimized reproduction more than an unsubstantiated theory. Learn PyTorch’s test conventions before proposing an optimization. Next, place PyTorch in the wider deployment ecosystem without turning the curriculum into shallow framework tourism in [Module 06](06-framework-landscape-and-interoperability.md).
