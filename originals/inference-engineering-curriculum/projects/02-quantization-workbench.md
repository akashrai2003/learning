# Capstone 02 — Quantization Workbench

## Objective

Build a reproducible system that converts quantization choices into measurable
operator error, model quality, memory footprint, and performance. The output is
a decision tool, not one “best” quantized checkpoint.

## Required quantizers

Implement or emulate:

- symmetric and asymmetric affine quantization;
- per-tensor, per-channel, and grouped scales;
- static and dynamic activation quantization;
- INT8 and packed INT4 weight-only paths;
- one second-order PTQ method such as a small GPTQ implementation;
- activation-aware scaling/search in the spirit of AWQ;
- FP8 format/scaling emulation;
- block-scaled FP4/microscaling emulation;
- KV-cache quantization.

You may use existing libraries as comparison targets, but your reference math,
packing, dequantization, and error analysis must be your own.

## Workbench architecture

```text
calibration dataset
 → activation capture
 → quantizer/search
 → packed artifact + metadata
 → reference dequantized operator
 → optional optimized kernel
 → layer/model evaluation
 → report
```

Store format metadata explicitly:

- bit width and code interpretation;
- scale dtype and shape;
- zero-point policy;
- group axis and size;
- padding and packing order;
- original tensor shape/stride;
- accumulation and output dtype;
- kernel compatibility requirements.

## Experiments

### 1. Synthetic distributions

Compare Gaussian, uniform, heavy-tailed, sparse, and outlier-contaminated
tensors. Plot clipping versus rounding error. Show why a calibration statistic
that works for one distribution fails on another.

### 2. Layer sensitivity

For a small model, quantize one tensor or layer at a time. Measure:

- mean/max error and cosine similarity;
- output/logit error;
- KL divergence;
- perplexity;
- at least one task or behavioral evaluation.

Rank sensitive layers and test a mixed-precision policy.

### 3. Group-size trade-off

For INT4 and block-scaled formats, sweep group size. Account for scale metadata
and padding:

\[
\text{effective bits/weight}
= b_\text{code}
+ \frac{b_\text{scale}+b_\text{zero}}{G}
+ \text{padding overhead}.
\]

Plot quality and actual file/VRAM size, not nominal bit width only.

### 4. Performance reality

Compare:

- packed size;
- time to load;
- dequantize-then-GEMM;
- fused/reference quantized operator;
- production optimized kernel if the target supports it.

Include model-relevant prefill and decode shapes. Document any case where a
lower-bit representation is slower because of dispatch, unpacking, metadata, or
unsupported Tensor Core paths.

### 5. KV-cache study

Sweep context length and cache format. Separate cache memory saved from
attention latency and long-context quality. Test accumulated generation, not
only reconstruction of one cache tensor.

## Quality protocol

Predeclare:

- calibration corpus and sample count;
- evaluation corpus/tasks;
- tokenizer/model revisions;
- baseline dtype;
- seeds and decoding policy;
- acceptance thresholds.

Do not tune on the final evaluation set. Report multiple tasks if making a
general quality claim.

## Artifact format

Create a self-describing small artifact format or metadata manifest. Write a
round-trip test:

```text
float tensor → quantize → serialize → load → dequantize
```

Verify codes, scales, padding, shapes, and byte order. Then inspect an existing
format such as GGUF or compressed-tensors and map its metadata into your mental
model.

## Acceptance gate

Given a new linear layer and target GPU, produce a written recommendation that
covers representation, scale granularity, calibration, accumulation, packing,
kernel availability, expected bottleneck, quality validation, and fallback.
Implement the selected reference path and demonstrate one failure of a naive
alternative.

## Stretch work

- automated mixed-precision search;
- learned or non-uniform codebooks;
- outlier channels in higher precision;
- multimodal encoder quantization;
- MoE expert/router sensitivity;
- export to a real runtime format;
- native NVFP4 comparison on Blackwell.

