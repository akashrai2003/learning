# Module 22 — GPTQ and Approximate Second-Order PTQ

**Phase:** 4 — Quantization  
**Pacing:** Week 11, paired with Module 21; three 3-hour sessions plus a weekend solver study  
**Prerequisites:** Modules 18–21; grouped weight-only quantization, matrix calculus, Cholesky factorization, and calibration

## Why this matters

Naively rounding each weight minimizes scalar weight error but ignores how input activations combine errors. GPTQ uses calibration activations and approximate second-order information to quantize a layer sequentially, compensating unquantized weights after each decision. It is foundational both as a practical W4A16 format and as a model for reasoning about data-aware PTQ.

## Outcomes

You can:

- derive GPTQ’s layer-reconstruction objective and Hessian approximation;
- implement a small sequential second-order quantizer;
- explain damping, column ordering, blocking, and group quantization;
- separate the GPTQ algorithm from its serialized packing/kernel backend;
- evaluate reconstruction, model quality, quantization time, and inference speed.

## Mental model: quantize, then compensate

For one linear layer, let weights \(W\in\mathbb{R}^{N\times K}\) and calibration inputs be columns \(X\in\mathbb{R}^{K\times T}\). GPTQ approximately minimizes

\[
\mathcal{L}(\hat W)=\|WX-\hat WX\|_F^2.
\]

For a row of \(W\), the Hessian of this quadratic objective is proportional to

\[
H=2XX^\top\in\mathbb{R}^{K\times K}.
\]

If coordinate \(q\) is fixed to its quantized value, an Optimal Brain Surgeon-style update distributes its error over remaining coordinates \(F\). Using \(H^{-1}\),

\[
\Delta w_F=
-\frac{w_q-\hat w_q}{[H^{-1}]_{qq}}[H^{-1}]_{qF}.
\]

The precise sign depends on whether code stores quantization error as
\((w_q-\hat w_q)\) or its negative; test against a direct reconstruction objective rather than copying an update blindly.

This compensation makes the next rounding decision on an updated weight vector. GPTQ processes columns in blocks and uses a Cholesky representation of the inverse-Hessian information to avoid solving a dense system for every scalar.

## Numerical and algorithmic details

Calibration may leave \(XX^\top\) singular because \(T<K\), features are correlated, or some channels are dead. Add damping, commonly a fraction of the mean diagonal:

\[
H_\lambda=H+\lambda I,\qquad
\lambda=\delta\cdot\operatorname{mean}(\operatorname{diag}H).
\]

Too little damping makes Cholesky fail or updates explode; too much erases second-order information. Record \(\delta\), calibration tokens, dtype, and factorization failures.

Column ordering matters. “Activation order” prioritizes columns with large Hessian diagonal, but reordering must be represented in the packed checkpoint or undone before the kernel consumes it. Group size controls qparam sharing; static groups and ordering interact because a group’s min/max may depend on which columns are adjacent.

Most implementations quantize one Transformer block at a time, collect its inputs, quantize contained linears, run the quantized block to produce inputs for the next block, and release memory. This sequential procedure prevents storing all model activations but also propagates prior quantization error.

## Implementation ladder

### 1. Reference

Implement `inference-lab/reference/gptq.py` for a tiny \(N,K\le128\) layer. Form \(H\) in FP64, add damping, and compare:

1. independent grouped rounding;
2. sequential compensation using an explicit inverse;
3. a brute-force optimum for a very small 2-bit problem.

Assert the reconstruction loss after every update.

### 2. Optimized

Replace the inverse with Cholesky solves, quantize columns in blocks, and batch weight rows. Add dead-column handling and optional activation ordering. Measure solver time and peak memory as \(K\) grows.

### 3. Integrated

Quantize a small decoder sequentially, serialize bit width, group size, symmetry, damping, ordering/permutation, and packing version, then load it through an official GPTQ backend. Verify the engine’s kernel rather than assuming the `quant_method` name implies acceleration.

## Three 3-hour sessions

1. **Derive and validate:** derive \(H\) and the constrained update; compare it numerically with a direct least-squares solve.
2. **Build the solver:** implement damping, Cholesky/block processing, grouping, and ordering with deterministic tests.
3. **Model integration:** quantize several layers, inspect reconstruction/PPL sensitivity, then trace loader, repacker, and W4A16 kernel.

The weekend study compares calibration sizes, damping values, group sizes, and ordering using one variable at a time.

## Experiments and deliverables

Produce:

- `reference/gptq.py`;
- `tests/test_gptq_tiny.py`, including singular Hessians;
- `benchmarks/results/gptq-ablation.json`;
- `benchmarks/reports/gptq-quality-cost.md`;
- `notes/source_walks/gptq-checkpoint-dispatch.md`.

Report layer reconstruction error, logit KL, PPL/task delta, quantization wall time/peak memory, checkpoint/GPU bytes, and prefill/decode latency. Pin the exact GPTQ implementation because format conventions differ.

## Common failure modes

- Forming \(H\) along the wrong activation dimension.
- Inverting a poorly conditioned Hessian without damping.
- Treating a lower reconstruction MSE as guaranteed lower PPL.
- Combining activation ordering with group boundaries incorrectly.
- Losing or double-applying the column permutation.
- Evaluating the layer on the same calibration samples only.
- Calling any checkpoint with `gptq` metadata algorithmically equivalent.
- Claiming the paper’s speedup with a backend that silently repacks or falls back.

## Mastery gate

For a small layer and supplied calibration matrix, derive the Hessian, implement one quantize-and-compensate step, explain damping and ordering, and demonstrate an example where GPTQ beats independent rounding on held-out output reconstruction. Then identify the exact physical format and runtime kernel used.

## Canonical code paths and OSS bridge

Read the [GPTQ paper](https://arxiv.org/abs/2210.17323) beside the authors’ [`IST-DASLab/gptq`](https://github.com/IST-DASLab/gptq) implementation. Inspect the layer Hessian accumulation, `fasterquant` update, packing code, and CUDA inference path. Then trace a production engine’s GPTQ config, weight loader, permutation handling, and kernel dispatch. High-value OSS work includes singular-Hessian tests, format validation, reproducible calibration scripts, and clear act-order fallback messages.

## Primary sources

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [Official GPTQ implementation](https://github.com/IST-DASLab/gptq)
- [Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)
- [vLLM GPTQ integration](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization)

## Next

Module 23 studies AWQ, which uses activation statistics to protect salient channels through equivalent scaling instead of second-order sequential compensation.
