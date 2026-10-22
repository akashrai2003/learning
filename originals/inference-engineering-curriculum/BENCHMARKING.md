# Benchmarking, Profiling, and Correctness Protocol

An optimization claim has four parts:

1. the semantics and quality that were preserved;
2. the workload and service objective;
3. the hardware/software environment;
4. the measured change with uncertainty.

“Tokens per second increased” is incomplete unless the report says whose
tokens, at what concurrency and lengths, under which latency constraints.

## Metric definitions

For request \(r\), let arrival be \(a_r\), first output token be \(f_r\), final
completion be \(c_r\), and emitted output-token count be \(O_r\).

- **TTFT:** \(f_r-a_r\), time to first token. It includes queueing and prefill.
- **E2E latency:** \(c_r-a_r\).
- **ITL:** intervals between successive streamed tokens.
- **TPOT:** commonly \((c_r-f_r)/(O_r-1)\) when \(O_r>1\). State the exact
  implementation used.
- **Request throughput:** completed requests per second.
- **Input throughput:** successfully processed prompt tokens per second.
- **Output throughput:** successfully generated output tokens per second.
- **Per-user decode rate:** output tokens divided by each request's decode
  interval; do not confuse it with aggregate output throughput.
- **Goodput:** completed work per time that meets defined TTFT/TPOT/E2E service
  objectives.

Always report distributions for latency: at least median, p90, and p99. Means
can hide queueing tails.

## Five benchmark levels

### 1. Microkernel

Question: is one kernel correct and efficient for its shape/dtype/layout space?

Record:

- full shape, strides, alignment, dtype, and accumulation dtype;
- warm-up and repeat count;
- median and dispersion of device-event time;
- bytes, FLOPs, achieved bandwidth/FLOPs, and roofline position;
- register/shared-memory usage, occupancy limiters, and relevant cache behavior;
- comparison to a strong library baseline, not only a Python implementation.

A fast square GEMM does not establish a fast decode GEMV-like shape.

### 2. Operator or block

Question: does fusion, quantization, or a new attention implementation improve a
real model shape while preserving output?

Include dispatch overhead, metadata/scales, layout conversions, and epilogues.
Measure a distribution of shapes taken from an actual model trace.

### 3. Offline engine

Question: what is maximum throughput when all work is available immediately?

Report batch/concurrency policy, prompt/output lengths, cache state, CUDA graph
state, and memory headroom. Offline throughput is not interactive serving
performance.

### 4. Online serving

Question: how does the system behave under arrival processes and SLOs?

Control:

- request-rate distribution and burstiness;
- concurrency cap;
- prompt/output length distribution and correlations;
- shared-prefix distribution;
- cancellation and streaming behavior;
- multimodal item sizes, if applicable;
- SLO thresholds.

Sweep load through saturation. One “infinite rate” run reveals capacity but not
the latency-throughput curve.

### 5. Quality and application behavior

Question: did the optimization preserve useful behavior?

Use multiple levels:

- operator error;
- layer/block output error;
- logits KL divergence or rank changes;
- perplexity on a held-out corpus;
- task evaluation appropriate to the model;
- targeted long-context, tool-calling, structured-output, and multimodal tests.

Never select calibration and evaluation examples from exactly the same sample.

## Required experiment manifest

Every result directory should contain:

```text
manifest.json          hardware, software, model and Git revisions
command.txt            exact server and client commands
workload.json          lengths, arrival process, seed, SLOs
raw/                   unmodified tool output and traces
summary.json           parsed metrics
report.md              hypothesis, result, interpretation, limitations
```

The report must state:

- baseline and treatment differ by which variables;
- predicted bottleneck and expected metric movement;
- warm-up and cache policy;
- number of independent runs;
- whether clocks/power were controlled or merely observed;
- failed requests and retries;
- peak host RAM and VRAM;
- quality acceptance threshold;
- known confounders.

## Workload matrix for the local 16 GB GPU

Begin with a model that leaves safe KV/activation headroom. Use fixed output
lengths (`ignore_eos`) only for controlled engine comparisons; use natural EOS
for application realism.

| Workload | Input/output tokens | Concurrency | What it stresses |
|---|---:|---:|---|
| Short interactive | 128 / 128 | 1, 2, 4, 8 | launch overhead, decode rate, queueing |
| Standard chat | 1,024 / 256 | 1, 2, 4, 8 | mixed prefill/decode |
| Long prefill | 8,192 / 128 | 1, 2 | attention prefill and chunking |
| Long decode | 128 / 1,024 | 1, 2, 4 | weight bandwidth and KV growth |
| Shared prefix | 4,096 shared + 256 unique / 128 | 2, 4, 8 | prefix-cache hit behavior |
| Bursty online | sampled realistic lengths | rate sweep | tail latency and goodput |

For each workload, first measure concurrency 1. Then increase concurrency until
throughput flattens or an SLO/memory limit is reached.

## Statistical practice

- Use at least one warm-up phase that exercises all measured shapes.
- Run enough measured requests to stabilize tail percentiles; 16 requests do
  not estimate p99.
- Repeat complete runs, not just internal kernel loops.
- Report median across runs and the range or confidence interval.
- Randomize treatment order when thermals or background activity could drift.
- Reject a “speedup” smaller than normal run-to-run variation.
- Preserve raw results, including unfavorable ones.

For deterministic kernels, randomized property tests still need fixed seeds.
For stochastic generation, compare distributions or task metrics and record
sampling parameters.

## Profiling ladder

Use the least invasive tool that can answer the current question.

1. **Application metrics/logging:** identify prefill, decode, queue, cache, and
   communication symptoms.
2. **PyTorch profiler:** find framework/operator costs and shapes.
3. **Nsight Systems:** locate CPU gaps, synchronization, copies, streams,
   collectives, graph launches, and dominant kernels on a timeline.
4. **Nsight Compute:** inspect one selected kernel's memory, instruction,
   occupancy, warp-stall, and roofline metrics.
5. **Generated code/PTX/SASS inspection:** explain the metric with actual
   instructions and resource use.

Start with Nsight Systems before collecting every Nsight Compute metric.
Profiler replay and metric collection perturb execution; never mix profiled
latency with normal serving latency.

Useful official references:

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## Correctness gates before performance gates

For a kernel PR:

1. compare against a high-precision oracle;
2. test supported and rejected dtype/shape/layout combinations;
3. test tail tiles and non-multiple dimensions;
4. test non-contiguous or explicitly reject them;
5. test extrema, NaN/Inf policy, and zero scales where relevant;
6. run memory/race tooling where supported;
7. test eager and compiled/custom-op paths;
8. test fallback dispatch;
9. run model-level generation/quality checks;
10. only then establish speed and regression thresholds.

For lossy methods, replace “bitwise equivalent” with a predeclared error and
quality budget. Do not move the budget after observing the result.

## Interpreting common outcomes

| Observation | Possible explanation to test |
|---|---|
| VRAM falls but latency does not | Kernel unavailable, dequant overhead, unchanged bottleneck, layout conversion |
| Batch-1 improves but aggregate regresses | Kernel shape specialization, scheduler or launch overhead, reduced batching efficiency |
| Kernel time improves but E2E does not | Operator not dominant, CPU/queue/communication bottleneck |
| Average improves but p99 worsens | unfair scheduling, cache misses, compilation, preemption, long-prefill interference |
| High occupancy but poor speed | occupancy is not utilization; memory dependencies or instruction mix may dominate |
| More concurrency lowers per-user tokens/s | throughput/latency trade-off, bandwidth sharing, queueing |
| Quantized perplexity is fine but tasks regress | outlier/token distribution, decoding sensitivity, calibration mismatch |

## vLLM benchmark baseline

Use the exact CLI exposed by the installed vLLM version. A representative
controlled run is:

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --model MODEL_ID \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 128 \
  --request-rate inf \
  --max-concurrency 1 \
  --ignore-eos \
  --seed 1 \
  --save-result \
  --save-detailed
```

Then sweep concurrency and finite request rates. The
[vLLM benchmark documentation](https://docs.vllm.ai/en/latest/cli/bench/serve.html)
defines its supported metrics. The
[DistServe paper](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
is a primary source for joint TTFT/TPOT goodput reasoning.

## Optimization report decision

End each report with one of:

- **Adopt:** correctness and quality pass; target workloads improve beyond
  uncertainty with no unacceptable regression.
- **Specialize:** improvement is real only for a declared dispatch region.
- **Revise:** hypothesis remains plausible but implementation or evidence is
  incomplete.
- **Reject:** bottleneck did not move as predicted or trade-offs exceed the
  budget.

“Interesting result” is not a deployment decision.

