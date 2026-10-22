# Benchmark Report — TITLE

**Date:**  
**Baseline commit/config:**  
**Treatment commit/config:**  
**Environment manifest:**  
**Raw results:**  

## Decision

Adopt / specialize / revise / reject.

## Hypothesis

Name the expected limiter, mechanism, and metrics that should move.

## Correctness and quality budget

Define oracle, test matrix, tolerances, model/task checks, and acceptance
thresholds before performance results.

## Workload

Record model/tokenizer revisions, input/output distributions, concurrency,
arrival process, sampling, prefix/multimodal properties, warm-up, cache state,
and SLOs.

## Environment

Summarize GPU/CPU/RAM, power/clocks, driver, toolkit, framework, runtime,
kernel/compiler versions, build flags, and important environment variables.

## Method

Commands, run count, ordering/randomization, timing method, profiler overhead,
and known confounders.

## Results

| Workload | Metric | Baseline | Treatment | Delta | Dispersion/CI |
|---|---|---:|---:|---:|---:|

Include correctness/quality, latency distribution, throughput/goodput, memory,
compile/startup behavior, and energy when relevant.

## Profiler evidence

Link the Nsight Systems timeline or operator trace first, then selected Nsight
Compute/generated-code evidence.

## Interpretation

Did the predicted bottleneck move? Why does the result vary by shape or load?

## Regressions and limits

List losing shapes, unsupported cases, fallback, portability, numerical
differences, and evidence still missing.

## Reproduction

Provide exact commands and artifact paths.

