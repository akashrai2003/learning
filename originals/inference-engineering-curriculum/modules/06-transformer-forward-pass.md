# Module 06 — The Transformer Forward Pass

> **Phase:** 2 — Architecture Core  
> **Time:** Week 3; two 3-hour sessions plus a 2–4 hour decoder lab  
> **Prerequisites:** Modules 03–05 and familiarity with probability distributions

## Why this matters

An engine can only optimize a model it executes exactly. “Transformer” names a family, not one computation: normalization placement, bias, QKV layout, positional method, head grouping, activation, residual scaling, and tied embeddings all vary by checkpoint. This module builds a small, auditable decoder whose tensor contracts become the correctness oracle for later kernels.

## Outcomes

You can:

- derive a decoder-only forward pass from configuration and weight shapes;
- track tensor shapes through embedding, attention, FFN, final norm, and LM head;
- explain why prefill and decode share semantics but use different state and kernels;
- separate architecture parameters from serving parameters;
- compare intermediate tensors against a canonical implementation.

## Mental model: residual stream plus two learned updates

For a common pre-normalized decoder layer:

\[
\begin{aligned}
r' &= r + \mathrm{Attention}(\mathrm{Norm}_1(r)),\\
r_{\text{next}} &= r' + \mathrm{FFN}(\mathrm{Norm}_2(r')).
\end{aligned}
\]

The residual stream \(r:[B,S,D]\) is the long-lived carrier. Attention mixes information across token positions; the FFN transforms each token independently. After \(L\) layers, a final norm and vocabulary projection produce logits.

Prefill evaluates \(S\) new positions and records their keys and values. Decode evaluates the newest position \(S_q=1\), attends to cached \(S_k\) positions, and appends one KV entry per layer. The mathematical output for the same prefix should agree; only organization and state reuse differ.

## Theory, math, and shapes

Let token IDs be `[B,S]` and embedding table \(E:[V,D]\). Gathering gives \(X:[B,S,D]\).

For one layer, a simple multi-head form uses:

\[
Q=XW_Q,\quad K=XW_K,\quad V_v=XW_V,
\]

initially `[B,S,H*D_h]`, then reshaped to `[B,H,S,D_h]`. Scaled causal attention is:

\[
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}+M\right),\qquad
O=AV_v,
\]

where \(M_{qk}=0\) for permitted keys and \(-\infty\) otherwise. Merge heads back to `[B,S,H*D_h]`, then project with \(W_O\) to `[B,S,D]`.

A generic gated FFN is:

\[
\mathrm{FFN}(x)=
\left[\phi(xW_{\text{gate}})\odot(xW_{\text{up}})\right]W_{\text{down}},
\]

with `[B,S,D] → [B,S,F] → [B,S,D]`. Module 09 studies its variants.

After the final norm, \(h:[B,S,D]\) maps through \(W_{\text{lm}}:[V,D]\):

\[
\text{logits}_{bsv}=\sum_d h_{bsd}W_{\text{lm},vd}.
\]

Generation commonly computes only `[B,V]` for selected positions. Training-style `[B,S,V]` can waste substantial memory.

## Implementation ladder

1. Create immutable `reference/config.py` fields for \(V,D,L,H,D_h,F\), epsilon, bias flags, activation, and weight tying. Validate divisibility and shape relations.
2. Implement `reference/decoder.py` from explicit PyTorch operations: embedding, normalization hooks, causal attention, FFN, residual updates, final norm, and LM head.
3. Keep `prefill(input_ids)` initially stateless and return logits plus per-layer K/V.
4. Add `decode_one(token_id, cache, position)` using concatenated reference cache. Optimize nothing yet.
5. Add forward hooks or named intermediate outputs so tests can compare layer-by-layer.
6. Add `tests/test_decoder.py`: full-prefix final logits must match iterative cached decode under FP32 tolerance.
7. Add weight-shape validation before execution. Report the first mismatched parameter by canonical name.

## Two 3-hour sessions

**Session A — derive and implement**

- **0:00–0:40 — Architecture derivation:** turn a model config into a shape table and parameter count.
- **0:40–1:10 — Block walk:** trace one token and one prompt through attention and FFN.
- **1:10–1:20 — Break.**
- **1:20–2:05 — Reference lab:** implement one layer and expose named intermediates.
- **2:05–2:35 — Stateful lab:** implement prefill and one-token decode with a simple cache.
- **2:35–3:00 — Parity lab:** compare full-prefix versus cached logits and inspect the first divergence.

**Session B — load and verify**

- **0:00–0:45:** write explicit weight-name and shape adapters for a tiny checkpoint.
- **0:45–1:30:** compare named intermediates against the canonical model.
- **1:30–2:15:** run full-prefix/cached parity across the test matrix.
- **2:15–3:00:** attach FLOP/byte/timing observations and document architecture-specific choices.

## Experiments and deliverables

Submit the configurable decoder, tests, a shape/parameter manifest, and `notes/forward-pass.md`. Use tiny unequal dimensions that are valid but reveal mistakes. Test:

- batch sizes 1 and 3;
- prompt lengths 1 and 7;
- full forward versus cached decode;
- tied versus untied LM head;
- all-masked invalid rows rejected explicitly.

Instrument each sublayer with estimated FLOPs, bytes, and elapsed time. Relate observations to Module 05 without optimizing yet.

## Failure modes

- Loading a checkpoint because parameter counts match while axis conventions differ.
- Applying normalization or residual addition in the wrong order.
- Scaling attention by \(\sqrt{D}\) instead of \(\sqrt{D_h}\).
- Building a mask with reversed query/key axes or wrong offset during decode.
- Returning logits for all prompt positions when only selected positions are needed.
- Enabling dropout or training behavior during inference.
- Mutating cache positions before a step succeeds.
- Comparing only sampled tokens; argmax can hide large logit errors.

## Mastery gate

Given only config and weight names, produce the forward graph, parameter shapes, and per-layer cache shape. Full-forward and iterative-decode final logits must agree, intermediate comparisons must identify an injected transpose bug, and invalid configs must fail before allocating model weights.

## Primary sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118)
- [Hugging Face Transformers model architecture source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [nanoGPT model reference](https://github.com/karpathy/nanoGPT/blob/master/model.py)

## Code paths to read

- Transformers: `src/transformers/models/llama/modeling_llama.py`
- nanoGPT: `model.py`
- vLLM: `vllm/model_executor/models/llama.py`
- vLLM: `vllm/model_executor/model_loader/`

## OSS bridge

Map your neutral config fields to one small Hugging Face causal LM. Load its state dictionary into an explicit name/transpose adapter, then compare embedding output, layer outputs, final hidden state, and logits. Keep the adapter outside `reference/decoder.py`; framework quirks must not contaminate the oracle.

## Next

Module 07 opens the boundaries before and after this decoder: tokenizer semantics, embeddings, logits, and reversible text generation.
