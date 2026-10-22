# Module 19 — Calibration, Error Metrics, and Evaluation

**Phase:** 4 — Quantization  
**Pacing:** Week 10, paired with Module 20; three 3-hour sessions and a weekend evaluation report  
**Prerequisites:** Module 18 plus language-model loss, dataset sampling, and reproducible benchmarking

## Why this matters

Quantization is an empirical approximation problem. A low layer MSE can coexist with worse perplexity, a stable perplexity can hide task regressions, and a calibration corpus that is too easy or too short can produce scales that fail in production. The engineer needs a chain of evidence from tensor statistics to logits to model quality and system performance.

## Outcomes

You can:

- design representative, leakage-free calibration samples;
- implement min/max, percentile, and error-search clipping;
- distinguish local reconstruction metrics from end-to-end quality;
- build a reproducible baseline-versus-quantized evaluation;
- diagnose which layer, token region, or activation channel caused a regression.

## Mental model: calibration chooses an operating range

For observed tensor \(x\), calibration selects parameters \(\theta=(s,z,c)\) for a quantizer \(Q_\theta\). Min/max uses extrema; percentile calibration chooses clipping bounds from quantiles; an MSE search solves approximately

\[
\theta^\star=\arg\min_\theta
\frac{1}{n}\sum_i\left(x_i-Q_\theta(x_i)\right)^2.
\]

Clipping trades rare large errors for a smaller step size on common values. For activation-aware weight quantization, the more relevant local objective for
\(X\in\mathbb{R}^{T\times K}\), \(W\in\mathbb{R}^{N\times K}\) is

\[
\|XW^\top-X\hat W^\top\|_F^2,
\]

not merely \(\|W-\hat W\|_F^2\). The calibration distribution therefore matters.

Use several complementary metrics:

\[
\operatorname{MSE}=\frac1n\|x-\hat x\|_2^2,\qquad
\operatorname{SQNR}=10\log_{10}\frac{\|x\|_2^2}{\|x-\hat x\|_2^2},
\]

\[
\operatorname{cos}(x,\hat x)=
\frac{x^\top\hat x}{\|x\|_2\|\hat x\|_2}.
\]

For logits \(l,\hat l\in\mathbb{R}^{V}\), compare stable softmax distributions with

\[
D_{\mathrm{KL}}(p\|\hat p)=
\sum_{v=1}^{V}p_v(\log p_v-\log\hat p_v),
\quad p=\operatorname{softmax}(l).
\]

Also record top-1 agreement, top-\(k\) set overlap, and chosen-token log-probability delta. These reveal decision-boundary changes that cosine similarity can miss.

Language-model perplexity for \(N\) evaluated target tokens is

\[
\operatorname{PPL}=\exp\left(-\frac1N\sum_{t=1}^{N}
\log p_\theta(x_t\mid x_{<t})\right).
\]

Report both baseline PPL and \(\Delta\)PPL using identical tokenization, stride, masks, and denominator. For long documents, prevent scoring the same overlap tokens twice.

## Calibration design

Build samples from the intended domain and length distribution. Preserve chat templates, code indentation, multilingual text, or multimodal preprocessing when those occur in service. Record sample count, token count, sequence-length histogram, seed, dataset revision, preprocessing code, and whether labels overlap an evaluation set.

Do not ask “is 128 samples enough?” without a convergence experiment. Recalibrate at 32, 128, 512, and 2048 sequences, then plot scale stability and downstream metrics. Outlier-sensitive schemes often stabilize differently by layer.

Create a sensitivity sweep: quantize one layer or operator family at a time, evaluate logits on a fixed probe set, and rank layers by KL or output reconstruction error. This supports mixed-precision decisions without guessing.

## Implementation ladder

### 1. Reference

Implement observers in `inference-lab/reference/calibration.py`: exact min/max, histogram/percentile clipping, and grid-search MSE clipping. Store counts and histograms, not entire activation corpora. Add hooks that capture layer input/output samples under `torch.inference_mode()`.

### 2. Optimized

Use streaming histograms or sketches, merge observer state deterministically across batches/ranks, and vectorize clipping search. Bound memory explicitly. Validate that distributed reduction produces the same qparams as a single-process run within tolerance.

### 3. Integrated

Create `engine/quantization/evaluate.py` that loads a pinned baseline and candidate, emits operator metrics, logit metrics, PPL/task metrics, memory, TTFT, TPOT, and throughput into one result schema. Acceptance gates must include quality and performance; neither substitutes for the other.

## Three 3-hour sessions

1. **Calibration theory:** visualize heavy-tailed tensors; implement and compare min/max, 99.9th percentile, and MSE clipping.
2. **Evaluation harness:** add layer hooks, logit KL/top-\(k\) agreement, and a sliding-window PPL evaluator with token-count assertions.
3. **Ablation:** vary calibration corpus, size, length, and seed one factor at a time; inspect one production observer/calibrator implementation.

The weekend report must state a pre-registered acceptance threshold, show confidence or run-to-run variation, and explain at least one disagreement between a local metric and end quality.

## Experiments and deliverables

Commit:

- `reference/calibration.py`;
- `tests/test_calibration.py`, including constant and all-zero tensors;
- `benchmarks/manifests/calibration-<model>.yaml`;
- `benchmarks/results/calibration-sweep.json`;
- `benchmarks/reports/quantization-quality.md`;
- `notes/source_walks/calibration-to-checkpoint.md`.

Keep prompt generation temperature zero for deterministic logit comparisons. For sampled quality tests, fix seeds but repeat enough seeds to avoid treating one generation as evidence.

## Common failure modes

- Calibration/evaluation leakage.
- Comparing different tokenizers, chat templates, truncation, or model revisions.
- Averaging per-batch perplexities instead of accumulating total NLL and tokens.
- Collecting only short prompts when activation range changes with position or modality.
- Optimizing tensor MSE without checking logits and tasks.
- Using cosine similarity alone; it is insensitive to magnitude error.
- Claiming a speedup while calibration, conversion, or fallback behavior changed between runs.

## Mastery gate

Given two quantized candidates, design a blinded evaluation that can reject either for a numerical, quality, or systems reason. Explain sampling, metrics, thresholds, confounders, and how you would localize a failure.

## Canonical code paths and OSS bridge

Read PyTorch’s [`MinMaxObserver`](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver.html), [`PerChannelMinMaxObserver`](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.PerChannelMinMaxObserver.html), and fake-quant equation in [`FakeQuantize`](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html). Compare those with an engine’s offline quantizer and `lm-evaluation-harness` request construction. Good OSS contributions include token-count correctness tests, calibration manifests, observer merge tests, and regression datasets.

## Primary and authoritative sources

- [ACIQ: Analytical Clipping for Integer Quantization](https://arxiv.org/abs/1810.05723)
- [Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [GPTQ](https://arxiv.org/abs/2210.17323), for activation-weighted reconstruction context
- [EleutherAI language-model evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [PyTorch quantization observers](https://docs.pytorch.org/docs/stable/quantization-support)

## Next

Module 20 applies this harness to INT8 static/dynamic activation quantization and SmoothQuant’s exact reparameterization.
