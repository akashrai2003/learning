# Module 27 — Multimodal Models and Processors

**Phase:** 4 — PyTorch model engineering and optimization  
**Time:** 12 focused hours  
**Prerequisites:** Transformer internals, vision/audio tensor contracts, and Module 25–26.

## Existing inference curriculum bridge

This implements [15 — Multimodal inference](../../inference-engineering-curriculum/modules/15-multimodal-inference.md) and reinforces [06 — Transformer forward pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), [10 — MHA/masking](../../inference-engineering-curriculum/modules/10-mha-masking-and-causal-attention.md), [11 — RoPE](../../inference-engineering-curriculum/modules/11-positional-encoding-rope-and-scaling.md), [14 — Long context and memory](../../inference-engineering-curriculum/modules/14-long-context-and-memory.md), and [17 — Architecture forensics](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md).

## Why this matters and outcomes

Modern multimodal systems are systems of contracts: processor output, feature encoder, projector/resampler, token insertion policy, positional encoding, attention mask, generation cache, and postprocessor. A model can be mathematically sound while wrong because one image has 576 visual tokens instead of the processor’s expected 729, or because prompt placeholders and image order disagree.

You will be able to reconstruct a multimodal forward pass from shapes, build a small image-prefix model in plain PyTorch, inspect a real processor/model pair without relying on opaque convenience calls, and diagnose quality or memory changes caused by visual-token policy.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P27 — Multimodal models and processors**.

1. [PyTorch multimodal examples](https://pytorch.org/blog/) — learn the API or system contract.
2. [CLIP paper](https://arxiv.org/abs/2103.00020) — study the underlying design, paper, or specification.
3. [Torchvision transforms v2](https://docs.pytorch.org/vision/stable/transforms.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [TorchAO source](https://github.com/pytorch/ao) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Torchtune source](https://github.com/pytorch/torchtune) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Let an image encoder produce \(H_v\in\mathbb{R}^{B\times N_v\times D_v}\). A projector produces visual language-space tokens

\[
Z_v=H_vW_p+b_p\in\mathbb{R}^{B\times N_v\times D},
\]

and text embeddings are \(Z_t\in\mathbb{R}^{B\times N_t\times D}\). A prefix model may concatenate `[BOS, Z_v, text]`, giving sequence length \(N_v+N_t+1\); a cross-attention design instead keeps vision memory separate and computes \(\operatorname{softmax}(Q_tK_v^T/\sqrt d)V_v\). These choices alter KV cache growth, mask topology, RoPE/position assignment, and decode cost. Video makes \(N_v\) include time; audio makes feature-frame count part of the token budget.

The processor must specify deterministic ordering and all transformations: media decode, sample/frame selection, resolution policy, crop/patching, text template, special placeholders, truncation, padding, and metadata. Save its source revision/config with an example’s post-processor tensors. Treat images/audio as first-class request fields, not byte strings converted invisibly inside `forward`.

For a batch with variable media, padding is not just text padding. You need masks/lengths that prevent attention to padded visual/audio features and an admission policy that budgets the true per-request token count. A 16 GB local GPU can be limited by image token count and KV cache even when model weights fit. Do a token/memory estimate before increasing resolution or frames.

## Implementation ladder in `torch-lab/`

1. **Reference:** implement `torch-lab/multimodal/toy_image_prefix.py`: a small patch encoder, linear projector, causal decoder, explicit token positions, and a causal/padding mask. Print every `(B,N,D)` shape and compare a hand-assembled sequence to the module path.
2. **Optimized:** build `torch-lab/multimodal/processor_contract.py` around a pinned real processor/model ecosystem. Save golden processor tensors and add visual-token budgeting, batched padding, and an optional `torch.compile` correctness run.
3. **Integrated:** send the same request through local preprocessing and a server-compatible processor/model path. Record first-token latency, visual token count, prefill memory, output quality, and cache behavior for images, multi-image inputs, and an optional audio/video case.

## Three focused sessions

1. **Shapes and masks:** derive prefix and cross-attention paths on paper, then build the toy model and assert tokens/positions/masks at every boundary.
2. **Processor forensics:** trace a real image/text request from bytes to logits; save every intermediate tensor and deliberately swap image order, resize policy, and placeholder location.
3. **Systems behavior:** sweep resolution/frame count/media count and profile prefill/decode separately. Explain observed memory from token count, head count, layers, dtype, and cache policy.

## Experiments and deliverables

Write `processor_contract.md`, `goldens/` references, `token_budget.py`, `test_media_order.py`, and `multimodal_sweep.csv`. Manifest media hashes/licenses, decoder versions, model/processor revisions, image/audio policy, placeholders, tensor shapes/dtypes, attention masks, positions, truncation, output checks, and latency/memory metrics.

## Common failure modes

- Assuming model config implies the exact processor behavior.
- Concatenating visual tokens without updating causal positions/masks.
- Treating image order and prompt placeholder order as independent.
- Benchmarks that ignore processor/decode time and only time GPU forward.
- Scaling resolution without accounting for quadratic attention and KV cache effects.

## Mastery gate

Given a multimodal answer mismatch, localize it to processor, encoder/projector, token insertion, masks/positions, decoder, or sampling using stored goldens. Explain the exact token and memory delta when image resolution doubles, and show a batch-safe implementation for variable media counts.

## Primary sources, code paths, and OSS bridge

- [PyTorch `MultiheadAttention` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [Torchvision model and transform documentation](https://docs.pytorch.org/vision/stable/models.html)
- [Torchtune repository](https://github.com/pytorch/torchtune)
- [Flamingo paper](https://arxiv.org/abs/2204.14198)
- [LLaVA paper](https://arxiv.org/abs/2304.08485)

Read the pinned model’s processor/config/model source and its tests, then compare vLLM’s multimodal input/processor paths. Useful OSS work: a processor parity fixture, variable-media batching test, token-budget diagnostic, or clear unsupported-feature error. Establish semantic parity before optimizing a vision encoder kernel.

## Next

Module 28 gives the experiments in this phase a quality, reproducibility, and systems-evidence discipline.
