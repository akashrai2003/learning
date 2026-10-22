# Module 28 — Evaluation, Quality, and Experiment Tracking

**Phase:** 4 — PyTorch model engineering and optimization  
**Time:** 9 focused hours  
**Prerequisites:** basic statistics, testing, and any prior model/optimization module.

## Existing inference curriculum bridge

Use [19 — Calibration, error metrics, and evaluation](../../inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md), [28 — Quantized kernel and system integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), [32 — Decoding](../../inference-engineering-curriculum/modules/32-decoding-sampling-and-logits-processing.md), [44 — Profiling, benchmarking, and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md), and the [benchmark report template](../../inference-engineering-curriculum/templates/BENCHMARK_REPORT.md).

## Why this matters and outcomes

Inference engineering is an evidence discipline. A throughput graph without prompt distribution, warmup, concurrency, quality constraint, model revision, and hardware state is a story, not a result. Likewise, a benchmark score without failure slices can hide tokenizer changes, prompt leakage, decode settings, or numerical regressions.

You will build a compact, local-first experiment ledger; select metrics that match the task; separate correctness, quality, and systems claims; use uncertainty where it matters; and make a result independently rerunnable. Optional trackers such as TensorBoard, MLflow, or Weights & Biases can help, but an immutable manifest and artifacts are the foundation.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P28 — Evaluation, quality, and experiment tracking**.

1. [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) — learn the API or system contract.
2. [EleutherAI evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) — study the underlying design, paper, or specification.
3. [MLflow tracking docs](https://mlflow.org/docs/latest/ml/tracking/) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [TorchAO source](https://github.com/pytorch/ao) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Torchtune source](https://github.com/pytorch/torchtune) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Every optimization sits on a Pareto surface: quality, latency, throughput, memory, cost, energy, reliability, and build complexity. State the non-negotiable constraint first—for example, “no more than 0.5 percentage-point accuracy loss,” “same tokenized prompt,” or “p99 TTFT below 500 ms”—then compare speed. For a mean metric \(\bar x\), bootstrap confidence intervals come from resampling the measured units (requests, examples, or trials) with replacement; do not report a tiny per-token standard deviation as independent evidence when requests share a batch or cache.

For classification track accuracy, per-class F1, calibration, and confusion slices. For generation record exact prompt/template, tokenizer, system/decode parameters, seed, EOS behavior, parsing rules, safety filters, and judge version; a single BLEU/LLM-judge number is insufficient. For vision/audio retain raw example IDs and preprocessing hashes. For numerical changes record output error distributions and task metric, not just one max difference.

Systems metrics must distinguish cold compile/load, warm single-request latency, TTFT, TPOT, aggregate output tokens/s, queue time, peak allocated/reserved memory, host RSS, and failure rate. A correct benchmark is a controlled experiment: one changed variable, stable power/clock where possible, a warmup policy, repeated trials, and raw data. Version every input artifact, including models, processors, datasets, calibration samples, and generated kernels.

## Implementation ladder in `torch-lab/`

1. **Reference:** implement `torch-lab/evaluation/manifest.py` and JSON Schema-like validation for `run.json`: git commit/diff, environment, hardware, model/weights/tokenizer hashes, data split hash, command, seed, metrics, and artifact paths.
2. **Optimized:** add `torch-lab/evaluation/runner.py` that runs fixed quality cases plus a systems harness, emits one JSONL record per request/trial, and computes bootstrap intervals from raw records. Mirror selected scalar metrics to a local tracker if desired.
3. **Integrated:** make the quantization, vision, multimodal, and serving labs consume the same manifest/evaluator. Add a regression threshold and a human-readable report that refuses to call speedup when parity or quality gates fail.

## Three focused sessions

1. **Metric design:** define task, unit, split, failure slices, quality floor, and systems metrics for one existing lab. Make a deliberately misleading aggregate and correct it.
2. **Reproducibility:** write/validate manifests, set deterministic controls where feasible, hash artifacts, and reproduce one run on a fresh process.
3. **Decision report:** run three repetitions of an optimization, bootstrap an interval, inspect outliers, and write a recommendation with explicit limits.

## Experiments and deliverables

Commit `schemas/run.schema.json`, `runs/*.json`, raw `requests.jsonl`, `evaluate.py`, `regression_policy.md`, and a report that includes both favorable and unfavorable results. Keep secrets/private prompts out of repository artifacts; store only approved hashes or redacted fixtures.

## Common failure modes

- Changing model revision, prompt template, and kernel at once.
- Treating GPU utilization as throughput or median as tail latency.
- Re-running until a favorable number appears and reporting only that run.
- Claiming deterministic output while sampling or nondeterministic kernels remain enabled.
- Sending sensitive production examples to a hosted experiment tracker by default.

## Mastery gate

Review an intentionally incomplete performance claim, list the missing evidence, and redesign it into a reproducible comparison. Then demonstrate that your harness catches a deliberately injected tokenizer, preprocessing, numerical, and latency regression.

## Primary sources, code paths, and OSS bridge

- [PyTorch reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [vLLM benchmarking documentation](https://docs.vllm.ai/en/latest/benchmarking/)
- [MLflow tracking documentation](https://mlflow.org/docs/latest/ml/tracking/)
- [NIST engineering statistics handbook](https://www.itl.nist.gov/div898/handbook/)

Read test/benchmark harnesses in PyTorch, vLLM, SGLang, TorchAO, and your target project. Valuable upstream contributions are stable reproductions, correct benchmark metadata, regression tests, quality fixtures, and analysis of a negative result. Never hide a correctness compromise behind a performance headline.

## Next

Module 29 begins the compiler track: FX makes many of these model and export contracts visible as graphs.
