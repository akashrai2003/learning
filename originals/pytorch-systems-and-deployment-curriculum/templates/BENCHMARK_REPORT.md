# Benchmark Report

## Claim

State one falsifiable claim. Example: “For batch 8, sequence lengths 256–2048,
and BF16 on this GPU, variant B reduces median decode-step latency without
changing output tolerance beyond X.”

## Workload and correctness

- Model/operator revision:
- Inputs and shape distribution:
- Seed and nondeterminism policy:
- Reference/oracle:
- Output/task parity metric and tolerance:

## Environment

- Git commits / package versions:
- GPU, driver, CUDA, compiler, runtime/library versions:
- CPU/RAM/OS and power/clock policy where relevant:
- Relevant environment variables and build flags:

## Method

- Warm-up / iteration count / synchronization boundaries:
- Timing source and aggregation statistic:
- Concurrency, streams, CUDA Graph policy:
- Memory measurement method:
- Cold compile/load and warm steady-state policy:

## Results

| Variant | Correctness | Median | p95 | Throughput | Peak VRAM | Peak host RAM | Build/compile | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Reference | | | | | | | | |
| Candidate | | | | | | | | |

## Interpretation

State the predicted bottleneck, observed bottleneck, confidence/variance,
unsupported cases, and next falsifying experiment. Link raw data and traces.
