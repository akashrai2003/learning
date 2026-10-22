# Module 23 — AWQ: Activation-Aware Weight Quantization

**Phase:** 4 — Quantization  
**Pacing:** Week 12, paired with Module 24; three 3-hour sessions plus a weekend GPTQ/AWQ comparison  
**Prerequisites:** Modules 19–22; calibration, weight-only packing, SmoothQuant, and GPTQ

## Why this matters

AWQ observes that weight importance is better revealed by the activations multiplying those weights than by weight magnitude alone. It protects salient input channels through mathematically equivalent scaling, then applies hardware-friendly grouped weight quantization rather than irregular mixed precision. The method illustrates how an accuracy technique and a kernel-friendly format must be co-designed.

## Outcomes

You can:

- derive AWQ’s activation-aware scaling transformation;
- implement scale and clipping searches for grouped INT4;
- distinguish AWQ from SmoothQuant and GPTQ;
- preserve graph equivalence while folding scales;
- validate an AWQ checkpoint’s packing and kernel selection.

## Mental model: protect channels without changing bit width

For \(X\in\mathbb{R}^{T\times K}\), \(W\in\mathbb{R}^{N\times K}\), and positive diagonal \(S\),

\[
Y=XW^\top=(XS^{-1})(WS)^\top.
\]

The floating-point function is unchanged, but quantizing \(WS\) gives important input channels a larger magnitude and therefore finer effective resolution after the inverse scaling of \(X\). AWQ estimates importance from activation statistics such as

\[
a_j=\frac1T\sum_t|X_{t,j}|
\]

and searches a family of scales, often proportional to \(a_j^\alpha\) with normalization. It chooses \(\alpha\) and optional clipping to minimize activation-weighted output error,

\[
\|XW^\top-XS^{-1}Q(WS)^\top\|_F^2.
\]

The original work reports that a small fraction of salient weights/channels strongly affects quality. The deployable solution does not store that fraction in higher precision; equivalent channel scaling keeps a regular INT4 kernel format.

## AWQ versus neighboring methods

- **Naive groupwise INT4:** qparams depend only on weight values.
- **GPTQ:** sequentially compensates errors using approximate second-order activation information.
- **SmoothQuant:** primarily moves activation outliers into weights to enable W8A8.
- **AWQ:** scales activation-important weight channels before weight-only quantization.

All may end in similar-looking packed W4A16 tensors. The checkpoint algorithm, group metadata, permutations, and runtime layout still differ.

## Graph folding

At a standalone linear, runtime could compute \(XS^{-1}\), but an extra multiply is undesirable. Fold the inverse scale into the preceding normalization or another safe producer, and multiply every corresponding consuming weight by \(S\). Residual branches and shared activations make this nontrivial. If an activation feeds Q, K, V, and MLP projections, graph equivalence requires consistent treatment of all consumers at the chosen boundary.

For grouped INT4, scales typically have shape
\([N,\lceil K/G\rceil]\); the AWQ channel scale has shape \([K]\). First transform \(W_{:,j}\leftarrow s_j W_{:,j}\), then compute group qparams and pack. Do not confuse these two kinds of scales.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/awq.py`. Capture \(X[T,K]\), compute activation statistics, sweep \(\alpha\), transform \(X,W\), and verify FP64 equality before quantization. Quantize \(WS\) with the Module 21 oracle and score output reconstruction on held-out activations.

### 2. Optimized

Avoid materializing every candidate transformed model: vectorize the scale/clipping search per linear or block. Add group sizes 32, 64, and 128 and a search budget manifest. Cache only bounded activation summaries or samples.

### 3. Integrated

Fold selected scales through a small decoder block, serialize its official AWQ representation, and run an engine W4A16 backend. Test prefill and decode separately and record any load-time repacking. Compare with naive INT4 and GPTQ at the same bit width/group size.

## Three 3-hour sessions

1. **Derivation:** build a synthetic layer where a small-magnitude weight channel sees large activations; show why weight MSE ranks it incorrectly.
2. **Search and fold:** implement scale/clipping search, prove graph equivalence, and test multi-consumer/residual cases.
3. **Production integration:** quantize a small model, trace AWQ config to packed parameters and kernel, and profile decode/prefill.

The weekend report compares naive INT4, GPTQ, and AWQ using the same baseline, calibration/evaluation sets, group size, target runtime, and benchmark shapes.

## Experiments and deliverables

Commit:

- `reference/awq.py`;
- `tests/test_awq_equivalence.py`;
- `benchmarks/results/awq-alpha-clipping.json`;
- `benchmarks/reports/awq-vs-gptq.md`;
- `notes/source_walks/awq-pack-and-dispatch.md`.

Include scale distributions, clipping ratios, output MSE/SQNR, logits KL, PPL/task deltas, bytes, TTFT, TPOT, and throughput. Preserve the unsmoothed FP model for comparison.

## Common failure modes

- Scaling \(W\) without applying/folding \(S^{-1}\) to activations.
- Applying the channel scale to the output rather than input axis.
- Folding through one consumer while leaving siblings unchanged.
- Evaluating the scale search only on calibration data.
- Mixing AWQ channel scales with group dequantization scales.
- Comparing methods with different group sizes or unquantized layer sets.
- Assuming `awq` selects the same kernel and layout in every runtime.

## Mastery gate

Construct an outlier-channel example, derive and implement AWQ’s equivalent transform, select \(\alpha\) on calibration data, demonstrate held-out reconstruction improvement over naive INT4, and prove the integrated graph still matches before quantization. Identify the exact production kernel and format.

## Canonical code paths and OSS bridge

Read the [AWQ paper](https://arxiv.org/abs/2306.00978) with the official [`llm-awq`](https://github.com/mit-han-lab/llm-awq) repository. Inspect activation collection, auto-scale, auto-clip, model-specific scale folding, TinyChat packing, and kernels. Compare vLLM’s [quantization integrations](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization) and TensorRT-LLM’s AWQ export path. OSS opportunities include new-model scale-folding tests, format assertions, and fair shape benchmarks.

## Primary sources

- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [Official AWQ and TinyChat implementation](https://github.com/mit-han-lab/llm-awq)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [GPTQ](https://arxiv.org/abs/2210.17323)

## Next

Module 24 replaces uniform integer grids with codebooks such as NF4 and examines how GGUF turns tensor encodings plus metadata into a portable inference artifact.
