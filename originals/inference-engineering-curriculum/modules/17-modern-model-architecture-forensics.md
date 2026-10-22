# Module 17 — Modern Model Architecture Forensics

**Phase:** 3 — Transformer architecture  
**Pacing:** Week 9, paired with Module 18; three 3-hour sessions plus the shared weekend review  
**Prerequisites:** Modules 06–16, especially cached decoding, RoPE/GQA, MoE, multimodality, and hybrid sequence models

## Why this matters

An inference engineer rarely receives a clean diagram. More often the evidence is a `config.json`, checkpoint tensor names, custom modeling code, and an engine error. The job is to reconstruct the executable graph, predict its memory and kernel requirements, and identify where a generic runtime will need special handling. A model name is not an architecture specification: two “Llama-like” checkpoints can differ in head dimensions, tied weights, sliding windows, normalization placement, expert routing, or quantization metadata.

## Outcomes

By the end, you can:

- reconstruct a decoder graph from configuration and checkpoint shapes;
- distinguish parameter count, active parameters, KV-cache size, and FLOPs;
- identify attention, position, MoE, multimodal, and recurrent special cases;
- predict unsupported operators and likely runtime fallbacks;
- verify a production implementation rather than trusting configuration names.

## Mental model: triangulate three sources of truth

Treat architecture forensics as a three-way consistency check:

1. **Configuration** states intent: layer count, hidden size, heads, experts, windows, and rope settings.
2. **Checkpoint tensors** reveal reality: actual shapes, packing, tied or omitted tensors, and per-layer variation.
3. **Forward code** defines semantics: tensor transforms, cache layout, masks, routing, and residual order.

No single source is sufficient. Remote code can reinterpret fields; conversion can transpose or fuse weights; heterogeneous models can override dimensions per layer.

For a dense linear layer with stored weight \(W\in\mathbb{R}^{N\times K}\),

\[
X_{[B,S,K]}W^\top_{[K,N]}\rightarrow Y_{[B,S,N]},
\qquad |W|=NK.
\]

For GQA with \(H_q\) query heads, \(H_{kv}\) KV heads, and head dimension \(D\),

\[
Q:[B,S,H_q,D],\quad K,V:[B,S,H_{kv},D].
\]

An unquantized per-request KV cache across \(L\) uniform layers costs

\[
2LSH_{kv}D\cdot \text{bytes(dtype)}.
\]

Do not apply that formula blindly to hybrid models: sliding-window layers, latent attention, cross-attention, or layer-specific head dimensions require separate cache groups.

For top-\(k\) MoE with \(E\) experts, total expert parameters scale with \(E\), while token-level expert FLOPs scale with \(k\). Router logits have shape \([BS,E]\), selected expert IDs \([BS,k]\), and dispatch introduces sorting, padding, and communication costs that a parameter count hides.

## Forensic workflow

Start with an evidence table:

| Question | Config evidence | Tensor evidence | Code evidence |
|---|---|---|---|
| Are embeddings tied? | `tie_word_embeddings` | missing or shared `lm_head` | assignment or tie hook |
| MHA, GQA, or MQA? | query/KV head counts | Q/K/V projection rows | repeat/broadcast logic |
| Dense or MoE? | expert count/top-k | expert-indexed weights | router and dispatch |
| Full or local attention? | window/pattern fields | sometimes none | mask selection per layer |
| Quantized? | quantization config | packed weights/scales | quant method and kernel dispatch |

Then derive every projection. With \(d=4096\), \(H_q=32\), \(H_{kv}=8\), and \(D=128\), Q has 4096 rows, but K and V each have only \(8\cdot128=1024\). If the checkpoint disagrees, investigate tensor parallel sharding, fused QKV packing, or a nonstandard head dimension before loading anything.

Finally trace one prefill and one decode step. Record the exact positions at which RoPE is applied, new K/V are appended, masks are constructed, logits are sliced, and cache objects mutate. Decode equivalence—cached token-by-token logits matching full-prefix logits—is the strongest small test.

## Implementation ladder

### 1. Reference

Write `inference-lab/reference/architecture_probe.py`. It should load JSON and safetensor headers without instantiating the model, emit a tensor inventory, infer projection shapes, and calculate parameter/KV estimates. Represent the reconstructed graph as typed records rather than prose.

### 2. Optimized

Add streaming header inspection, fused-QKV recognizers, layer-pattern detection, and formulas for dense versus MoE active FLOPs. Generate separate prefill/decode shape tables and flag divisibility constraints such as \(H_q\bmod H_{kv}=0\).

### 3. Integrated

Compare the report with an actual runtime trace. Hook module inputs/outputs for a tiny prompt, then inspect the engine’s model registry, attention backend choice, cache specification, and quantization dispatch. The probe must fail clearly on unknown patterns rather than silently label everything “Llama.”

## Three 3-hour sessions

1. **Derive and inventory:** choose one dense and one MoE or hybrid checkpoint; map config fields to tensor shapes and hand-check ten parameter formulas.
2. **Implement and test:** build the probe and cached-versus-full-logit tests on a small model. Add intentionally inconsistent fixtures.
3. **Production source walk:** trace model registration through construction, attention selection, and cache allocation in one engine; annotate file paths and the pinned commit.

Use the weekend to produce a two-page architecture dossier containing graph, shapes, parameter/active-FLOP estimates, cache formula, unsupported features, and expected kernels.

## Experiments and deliverables

Commit:

- `models/architecture_notes/<model>.md` with an evidence table;
- `reference/architecture_probe.py`;
- `tests/test_architecture_probe.py`;
- `benchmarks/reports/<model>-prefill-decode-shapes.md`;
- `notes/source_walks/model-load-and-dispatch.md`.

Reproducibility requires model revision, tokenizer revision, trust-remote-code setting, framework/engine commits, and the raw configuration and tensor-name digest.

## Common failure modes

- Inferring semantics from a model family name.
- Counting all MoE experts as active for every token.
- Assuming `hidden_size / num_heads` is every attention head dimension.
- Ignoring tensor parallel shards, fused projections, tied weights, or quantizer side tensors.
- Treating multimodal preprocessing as outside the model graph.
- Testing only prefill; cache position bugs often appear only during decode.
- Executing untrusted remote modeling code merely to inspect a checkpoint.

## Mastery gate

Given an unfamiliar config and tensor manifest, produce within 45 minutes: the decoder block equation, Q/K/V and MLP shapes, total and active parameter estimates, per-token KV bytes, likely special operators, and three invariants to test. Defend every conclusion with config, tensor, or code evidence.

## Canonical code paths and OSS bridge

Read Hugging Face `configuration_*.py` beside `modeling_*.py`, especially [`modeling_mixtral.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py). In Megatron Core, inspect [`transformer_config.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py) and the [MoE implementation notes](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md). In vLLM, begin at the model registry, then follow the selected model class into attention and quantization configuration. A valuable first OSS contribution is a precise unsupported-config error, shape assertion, or architecture documentation test.

## Primary sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RoFormer / rotary position embedding](https://arxiv.org/abs/2104.09864)
- [Grouped-Query Attention](https://arxiv.org/abs/2305.13245)

## Next

Module 18 turns the tensor inventory into an explicit numerical representation: values, scales, zero points, groups, and packed storage.
