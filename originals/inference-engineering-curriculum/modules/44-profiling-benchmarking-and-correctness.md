# Module 44 — Profiling, Benchmarking, and Correctness

**Phase:** 6 — GPU, kernels, compilers, and builds  
**Pacing:** Week 22, paired with Module 43; three 3-hour sessions plus the Phase 6 gate  
**Prerequisites:** Modules 05, 19, 28, and 35–43; rooflines, numerical evaluation, engine metrics, asynchronous timing, and build provenance

## Why this matters

Performance work is an argument from evidence: the result is correct, the measured workload represents the claim, timing is synchronized, the intended kernel ran, the bottleneck mechanism changed, and regressions outside the winning shape were checked. A profiler screenshot or best-of-one tokens/s number cannot establish that argument. This module turns the learner's SM120 build into a reproducible investigation rather than an anecdote.

## Outcomes

You can:

- design oracle, operator, model-quality, microbenchmark, and serving tests as separate layers;
- use device events, Nsight Systems, Nsight Compute, NVTX, and engine metrics appropriately;
- calculate latency distributions, bandwidth, arithmetic intensity, and useful throughput;
- distinguish cold build/JIT/autotune/capture from warm steady state;
- define correctness and performance regression gates before optimizing.

## Mental model: descend, explain, then ascend

Begin with end-to-end symptoms, descend only as far as needed, and return to end-to-end validation:

```text
SLO/serving -> model phase -> operator -> kernel -> instruction/memory counters
       explanation and change
kernel -> operator parity -> model quality -> serving/SLO regression
```

At the operator level, compare with a trusted explicit implementation across shape, dtype, layout, tails, seeds, and streams. Use absolute/relative error, ULP where meaningful, NaN/Inf, and repeatability; choose tolerances from accumulation behavior rather than copying a default. At the model level, use logits/KL or top-\(k\) agreement plus PPL/tasks and cached decode. Sampling text equality is neither necessary nor sufficient for stochastic systems.

For a kernel processing \(W\) operations and \(Q\) measured bytes in time \(t\),

\[
P=W/t,\qquad B=Q/t,\qquad I=W/Q.
\]

Compare achieved performance with the relevant compute and hierarchical memory ceilings. Theoretical tensor bytes are not profiler-measured DRAM bytes when caches/reuse intervene. Occupancy, Tensor Core utilization, stalls, and cache metrics are clues that must support a mechanism, not scores to maximize independently.

At service level, report TTFT, TPOT/ITL, end-to-end latency, request/output-token throughput, errors, and goodput under a declared arrival process and concurrency. Include percentiles and raw samples. Batch-one decode and high-concurrency aggregate throughput answer different questions.

## Tools answer different questions

- **CUDA events/framework benchmark timers:** accurate focused device elapsed time.
- **Nsight Systems:** CPU/GPU timeline, launches, streams, gaps, copies, collectives, synchronization, and NVTX ranges.
- **Nsight Compute:** one/few kernels' instructions, occupancy, memory hierarchy, rooflines, and stalls.
- **PyTorch profiler/compiler traces:** framework operators, shapes, graph breaks, generated-code correlation.
- **Engine benchmark/metrics:** queueing, request tails, cache/speculation behavior, and goodput.

Profilers add overhead. Reproduce without profiling, capture a bounded representative range, and do not treat profiled wall time as production throughput. Save raw reports and exact commands.

## Implementation ladder

### 1. Reference

Create `inference-lab/benchmarks/harness.py` with environment manifests, warm-up, synchronization, independent repetitions, raw JSON, summary statistics, and correctness callbacks. Add seeded shape generators and a paired-baseline mode.

### 2. Optimized

Benchmark one Module 39–41 kernel. Confirm dispatch, use device timing, collect a Systems timeline, then a focused Compute report. Form one falsifiable hypothesis—for example, uncoalesced scale loads limit an NVFP4 tile—and change one variable.

### 3. Integrated

Serve the same pinned model with baseline and treatment on the RTX 5060 Ti. Benchmark prefill and decode separately, then arrival-rate/concurrency sweeps. Preserve cold startup/build/JIT/capture timings separately. Check logits/quality and memory before accepting throughput.

## Three 3-hour sessions

1. **Correctness and harness:** define contracts/tolerances, build raw-data capture, and test that an intentionally wrong fast kernel fails.
2. **Profiler investigation:** mark NVTX ranges, locate timeline gaps, select a hot kernel, and connect Compute counters to a roofline hypothesis.
3. **Serving experiment:** run paired workloads, calculate uncertainty and goodput, check quality/memory, and write a regression decision.

## Experiments and deliverables

Commit:

- `benchmarks/harness.py`;
- `tests/test_benchmark_oracles.py`;
- `benchmarks/manifests/sm120-engine.yaml`;
- `benchmarks/results/sm120-baseline-treatment.json`;
- `benchmarks/reports/sm120-investigation.md`;
- `notes/profiles/sm120-timeline-and-kernel.md`.

The report must include pinned commits/checkpoint, driver/toolkit/framework, clocks/power state, GPU exclusivity, command/flags, input and arrival distributions, warm-up, cache state, synchronization, repetitions, raw values, percentiles/dispersion, selected dispatch, quality, memory, profiler overhead, and limitations.

## Common failure modes

- Optimizing before a correctness oracle and regression threshold exist.
- Timing asynchronous launches with an unsynchronized CPU clock.
- Reporting the fastest trial instead of distributions.
- Mixing compilation/autotuning/graph capture into only one treatment.
- Comparing different prompts, output lengths, stopping rules, or arrival rates.
- Profiling the wrong process, rank, phase, or fallback kernel.
- Reading high occupancy as proof of efficiency.
- Using advertised peak TOPS or bandwidth as achieved performance.
- Claiming kernel speedup without end-to-end critical-path share.
- Reporting aggregate throughput while tail SLO goodput collapses.

## Mastery gate

Produce a reviewer-reproducible report for one optimization: a predeclared correctness gate, raw paired results, synchronized timing, Systems evidence locating the path, Compute evidence supporting the mechanism, and end-to-end prefill/decode/serving validation. Another learner must reproduce the conclusion from the manifest. Reject your own optimization if quality, memory, tails, or untested shapes violate the gate.

## Canonical code areas and OSS bridge

Use [vLLM benchmark CLI/source](https://docs.vllm.ai/en/latest/cli/bench/) and its [profiling guide](https://docs.vllm.ai/en/latest/contributing/profiling/), SGLang's `bench_serving.py` and developer benchmark guide, PyTorch benchmark utilities, and NVIDIA profiler docs. Reviewer-ready OSS work supplies the failing test or baseline, pinned workload matrix, raw data, profiler mechanism, quality checks, and stated limits; never attach only a dashboard screenshot.

## Primary and authoritative sources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [vLLM serving benchmark](https://docs.vllm.ai/en/latest/cli/bench/serve/)
- [vLLM profiling guide](https://docs.vllm.ai/en/latest/contributing/profiling/)

## Next

Module 45 expands the measurement model across ranks, where collective latency, topology, sharding, and synchronization join the critical path.
