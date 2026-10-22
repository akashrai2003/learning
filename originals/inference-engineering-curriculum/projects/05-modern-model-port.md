# Capstone 05 — Modern-Model Architecture and Runtime Port

## Objective

Select one recent open-weight model whose architecture exercises more than a
plain Llama block. Reconstruct it from primary artifacts, implement a small
reference path, and port or validate it in one target runtime.

Useful challenge features include:

- MoE routing or shared experts;
- sliding/global/hybrid attention;
- nonstandard RoPE or long-context scaling;
- latent, compressed, or unusual KV representation;
- multimodal encoder/projector/resampler;
- state-space/recurrent layers;
- FP8/FP4 or mixed quantization metadata;
- multi-token prediction or speculative heads.

Choose a model small enough to test locally or construct an official tiny/random
checkpoint where project conventions permit.

## Architecture dossier

Derive from the model card, configuration, tokenizer/processor files, checkpoint
tensor names, source implementation, and original paper:

- vocabulary and special-token behavior;
- hidden/intermediate dimensions and layer counts;
- attention head counts, head dimensions, and KV grouping;
- positional method;
- normalization and residual topology;
- FFN or expert topology and routing;
- multimodal interfaces;
- cache contents and growth;
- logits/sampling-related special behavior;
- supported dtypes and quantization recipe;
- parameter count, active parameters, FLOPs, and memory by component.

Resolve contradictions experimentally and record them.

## Reference slice

Implement the smallest feature that distinguishes the architecture:

- one hybrid attention block;
- one routed MoE block;
- one multimodal projector/resampler;
- one recurrent state update;
- one unusual quantized linear path.

Use checkpoint tensors or deterministic synthetic weights. Compare intermediate
outputs to the canonical implementation.

## Runtime port plan

Map the architecture into the target:

```text
configuration registration
 → processor/tokenizer
 → weight-name mapping
 → model class/graph construction
 → attention/cache backend
 → quantization/operator dispatch
 → sampler/output
 → tests and documentation
```

List unsupported operations and fallback behavior before optimization.

## Tests

Include:

- configuration construction;
- checkpoint weight coverage—missing and unexpected tensors;
- deterministic tiny forward outputs;
- cached versus uncached decode;
- dtype/device path;
- quantization metadata if relevant;
- multimodal placeholder/item alignment if relevant;
- tensor-parallel behavior analytically and, when accessible, empirically;
- generation smoke test;
- memory leak or cache-cleanup behavior where relevant.

Use the target project's model-test conventions.

## Performance dossier

Record:

- load and compile time;
- peak host RAM and VRAM;
- resident weights and cache capacity;
- prefill/decode profiles;
- selected and fallback kernels;
- single-request and concurrent metrics;
- architecture-specific bottlenecks;
- quality/sanity evaluation.

If local hardware cannot execute the full model, clearly separate tiny-model
correctness, analytical estimates, and any remotely obtained full-model
measurements.

## Acceptance gate

Explain the model from configuration to one generated token without relying on
its marketing diagram. Given a mismatched output, localize it through weight
mapping, processor/tokenizer, position/cache, block computation, and sampling.
Given a target runtime, identify the smallest well-tested patch sequence needed
for support.

## Stretch work

- upstream model support;
- optimized kernel for its novel operator;
- quantized variant;
- multimodal batching;
- distributed expert/context parallelism;
- speculative or multi-token-prediction integration.

