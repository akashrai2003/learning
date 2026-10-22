# Module 15 — Multimodal Inference

> **Phase:** 3 — Advanced Architectures  
> **Time:** Week 8; two 3-hour sessions plus a 2–4 hour integration lab and mastery gate  
> **Prerequisites:** Modules 03–14; tokenization/templates, attention and KV caching, schedulers, hashing, and latency measurement

## Why this matters

A vision-language request is not merely “text plus an image.” Media decoding, resizing, normalization, patchification, a modality encoder, a projector, placeholder expansion, and language-model prefill must agree with the checkpoint. Image resolution can add hundreds or thousands of effective tokens, changing TTFT, KV usage, batching, and admission. Serving remote media also creates security and reliability boundaries absent from an in-memory tensor example.

## Outcomes

You will be able to:

- trace raw media through preprocessing, encoding, projection, token merge, and decoding;
- derive visual-token and language-sequence shapes;
- distinguish early-fusion/projector models from cross-attention models;
- make multimodal cost visible to cache allocation and scheduling; and
- validate template, placeholder, preprocessing, cache identity, and input-security contracts.

## Mental model

Think of a multimodal model as a typed compiler pipeline. A raw image is decoded into pixels, compiled by a model-specific processor into a tensor, encoded into visual features, projected into the language model's hidden width, and spliced into reserved positions in the token stream. The decoder then sees vectors of a common width, but their origins still matter for position rules, masks, caching, and accounting. If any stage uses a different “dialect” from training, shapes may pass while answers degrade.

## Theory, math, and shapes

For images `I: [B,3,H,W]` and square patch size `P`, a simple vision transformer forms

$$
N=\frac{H}{P}\frac{W}{P}
$$

patch tokens, possibly plus class or special tokens. Flattened patches are `[B,N,3P²]`; a patch projection and vision encoder produce `V: [B,N,d_v]`. A connector maps modalities:

$$
Z=\operatorname{Projector}(V)\in\mathbb{R}^{B\times N\times d_m},
$$

where `d_m` is the language-model width. In an early-fusion LLaVA-style model, `N` placeholder positions are replaced by `Z`, creating a combined embedding sequence `[B,T_text+N,d_m]` (exact special-token accounting is model-specific). These visual positions participate in decoder prefill and usually produce KV entries like text positions.

Dynamic resolution, tiling, video frames, pooling, or patch merging makes `N` input-dependent. Never estimate it from file bytes. Query the processor/model contract and budget the post-processor token count.

Flamingo-style models instead keep visual latents separate and add cross-attention. Decoder self-attention cache and static visual K/V then have different lifecycles. Audio and video follow the same systems pattern but add sample-rate/frame-selection and temporal-shape contracts.

Total latency can be decomposed into fetch/decode, CPU processing, host-to-device transfer, encoder/projector, LM prefill, and decode. Reusing an identical image may avoid early stages through processor or encoder caching, but cache keys must incorporate media content/UUID, processor version and parameters, model weights, and relevant trust namespace.

## Implementation ladder

1. Add `inference-lab/reference/vision.py` with a deterministic patchifier and linear “toy encoder.” Assert channel order, divisibility/padding policy, patch order, and output token count.
2. Add `reference/multimodal_merge.py` that replaces an exact placeholder span with projected embeddings. Return a provenance map alongside the combined tensor.
3. Add `engine/multimodal.py` defining typed inputs, per-modality limits, post-processing token estimates, stable identities, and stage timings.
4. Extend `engine/scheduler.py` so admission uses text plus expanded media tokens and accounts for encoder work.
5. Add an optional real-model adapter only after synthetic invariants pass. Treat its processor and chat template as checkpoint assets, not generic defaults.

## Two 3-hour sessions

**Session A — model data path**

- 0:00–0:45: trace a LLaVA-style request and derive every shape.
- 0:45–1:30: implement deterministic decode assumptions, patchify, encode, and project.
- 1:30–2:15: implement placeholder merge and provenance tracking.
- 2:15–3:00: test variable resolution, multiple images, ordering, and text-only behavior.

**Session B — serving path**

- 0:00–0:40: define typed request limits and stable cache identity.
- 0:40–1:25: integrate media-token accounting with admission and paging.
- 1:25–2:10: instrument cold versus repeated-media stage latency.
- 2:10–3:00: batch variable media shapes and threat-model remote inputs.

## Experiments and deliverables

Commit:

- `tests/test_multimodal.py` covering patch order, placeholder count, multi-image order, text-only equivalence, and malformed embeddings;
- `benchmarks/bench_multimodal.py` reporting every latency stage, visual-token count, LM TTFT, and peak memory;
- a resolution/tiling sweep that plots visual tokens and TTFT;
- a cold-versus-warm repeated-image experiment with cache-hit verification;
- a batching experiment comparing equal-token bucketing with naive request-count batching; and
- `notes/module15.md` containing the pipeline contract and a concise media-ingress threat model.

For one tiny image, save pixels after preprocessing, patch order, projected embeddings, placeholder indices, and merged sequence length as a golden trace.

## Common failure modes

- Using a generic resize, crop, RGB normalization, or patch order instead of the checkpoint processor.
- A chat template emits one media marker while the model expects an expanded placeholder span.
- Counting images rather than expanded visual tokens for scheduling.
- Reordering interleaved images or pairing one image's metadata with another embedding.
- Applying text positions or causal masks incorrectly across inserted media.
- Re-encoding identical media every chat turn, or reusing cached features after processor/model changes.
- Accepting arbitrary precomputed embeddings from untrusted clients; wrong shapes or values bypass the encoder contract.
- Fetching unrestricted URLs, following redirects into private networks, or decoding oversized/decompression-bomb media.
- Batching highly variable resolutions without bucketing, causing padding and stragglers.
- Measuring only LM tokens/s while hiding encoder and CPU preprocessing latency.

## Mastery gate

Given image size, patch/merge configuration, special-token policy, and text length, derive the exact combined length and resulting per-layer KV increase. Your golden synthetic example must preserve patch and image order and reject one-too-few placeholders. A benchmark must separate cold processor/encoder time from LM prefill/decode and demonstrate that scheduler accounting prevents a media-heavy request from exceeding its token budget.

## Primary sources

- Radford et al., [“Learning Transferable Visual Models From Natural Language Supervision”](https://arxiv.org/abs/2103.00020) and the [official CLIP repository](https://github.com/openai/CLIP).
- Liu et al., [“Visual Instruction Tuning”](https://arxiv.org/abs/2304.08485) and the [official LLaVA repository](https://github.com/haotian-liu/LLaVA).
- Alayrac et al., [“Flamingo: a Visual Language Model for Few-Shot Learning”](https://arxiv.org/abs/2204.14198).
- vLLM, [multimodal input contract and serving examples](https://docs.vllm.ai/en/latest/features/multimodal_inputs/).

## Code paths to read

Your paths are `reference/vision.py`, `reference/multimodal_merge.py`, `engine/multimodal.py`, `engine/scheduler.py`, `tests/test_multimodal.py`, and `benchmarks/bench_multimodal.py`.

## OSS bridge

In LLaVA, locate the vision tower, projector, and image-token preparation before reading generation code. In vLLM, trace [`multimodal/processing/processor.py`](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/processing/processor.py), [`multimodal/registry.py`](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/registry.py), processor caches, and model-specific multimodal interfaces. Map each production object to one stage in your trace.

## Next

Module 16 replaces some or all growing attention history with recurrent state and extends the engine to SSM and hybrid cache lifecycles.
