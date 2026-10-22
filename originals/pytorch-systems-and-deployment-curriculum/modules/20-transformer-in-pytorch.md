# Module 20 — Transformer in PyTorch

> **Phase:** III — Model architecture and implementation  
> **Time:** 12–15 focused hours across four sessions  
> **Prerequisites:** Modules 08–19, linear algebra, autograd, and the minimal training engine

## Existing inference curriculum bridge

This is the executable companion to [Module 06 — Transformer Forward Pass](../../inference-engineering-curriculum/modules/06-transformer-forward-pass.md), [Module 10 — MHA, Masking, and Causal Attention](../../inference-engineering-curriculum/modules/10-mha-masking-and-causal-attention.md), [Module 12 — MQA, GQA, and KV Cache](../../inference-engineering-curriculum/modules/12-mqa-gqa-and-kv-cache.md), [Module 29 — Attention IO Complexity and Online Softmax](../../inference-engineering-curriculum/modules/29-attention-io-complexity-and-online-softmax.md), [Module 30 — FlashAttention Evolution](../../inference-engineering-curriculum/modules/30-flashattention-evolution.md), and [Module 31 — PagedAttention and Memory Management](../../inference-engineering-curriculum/modules/31-pagedattention-and-memory-management.md). Preserve this reference as later kernels and serving runtimes become less transparent.

## Why this matters and outcomes

You cannot optimize a Transformer you cannot rebuild and test. A compact decoder-only implementation exposes the interfaces that production implementations specialize: token IDs, embeddings, normalization, QKV projection/layout, causal attention, gated FFN, residual topology, logits, loss, and state. The goal is not to outgrow PyTorch’s library in one module; it is to own a correct oracle.

By the end, you can derive every major decoder tensor shape and parameter count; implement a configurable pre-norm decoder; use stateful modules and functional attention deliberately; test causal semantics and gradients; and compare a readable reference against a vectorized PyTorch implementation without changing the model contract.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P20 — Transformer in PyTorch**.

1. [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — learn the API or system contract.
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — study the underlying design, paper, or specification.
3. [PyTorch Transformer tutorial](https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch Transformer tutorial](https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and tensor shapes

Let B be batch, T tokens, D model width, H heads, Dh=D/H, F FFN width, and V vocabulary. input_ids is int64[B,T]. Token embedding produces X in float[B,T,D]; positional representation is added or applied according to the selected architecture. A pre-norm block can be written:

U = X + Attention(Norm(X))  
Y = U + FFN(Norm(U)).

A fused QKV projection maps [B,T,D] to [B,T,3D], then reshapes to Q,K,V each [B,H,T,Dh]. Causal attention computes

S[b,h,q,k] = Q[b,h,q,:] dot K[b,h,k,:] / sqrt(Dh),  
P = softmax(S + causal_mask),  
A[b,h,q,:] = sum_k P[b,h,q,k] V[b,h,k,:].

A causal mask forbids k>q. Merge heads to [B,T,D], apply output projection, and return to the residual stream. A SwiGLU-style FFN commonly uses gate and up projections [D,F], then down [F,D]:

FFN(x) = Wdown (silu(x Wgate) elementwise (x Wup)).

Exact normalization, bias, activation, positional encoding, and MLP expansion are architecture choices. Put them in immutable config and state their tensor contracts. Token embedding and LM-head weight tying means the same Parameter may serve lookup and logits: logits[B,T,V] = hidden[B,T,D] times Wembedᵀ[V,D]. Cross entropy consumes shifted label IDs with an explicit ignore index for padding.

Readability is part of correctness. Keep an explicit reference attention score path for tiny inputs, but use torch.nn.functional.scaled_dot_product_attention only after confirming its mask convention, dropout behavior, shape layout, and causal flag. SDPA can select optimized kernels; it preserves an attention contract, not a promise of a particular backend. Do not rely on a model sampling successfully as proof of causal correctness.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/models/decoder.py with config, token embedding, RMSNorm or LayerNorm, one pre-norm decoder block, explicit masked attention, gated FFN, logits, and cross-entropy loss.
2. Test literal tiny tensors, parameter counts, every shape/stride boundary, future-token noninterference, padding loss, train/eval dropout, and finite gradients.

**Optimized**

3. Replace only reference score/value attention with SDPA after exact parity tests on short sequences. Keep an implementation switch and report selected layout/mask/dtype constraints.
4. Batch/flatten compatible projections, preserve tied weights, and profile prefill separately from one-token decode. Explain every remaining quadratic allocation.

**Integrated**

5. Train the small decoder with Module 19’s engine, save/load its state, and implement deterministic greedy generation. Keep config/tokenizer/data assumptions in the artifact manifest.
6. Add a test suite that compares reference and optimized paths under random asymmetric B/T/H shapes, non-contiguous inputs where supported, and a fixed seed. Module 21 adds KV-cache decoding to this same tested model.

## Four focused sessions

1. **Architecture on paper (3h):** derive shapes, parameters, FLOPs, and residual order for one block; write the config and contract table first.
2. **Reference block (3h):** implement embedding, normalization, attention, FFN, loss, and tiny literal tests.
3. **Semantic tests (3h):** test causal leakage, shifted labels, tied weights, gradients, dropout modes, and state-dictionary round trip.
4. **Vectorized path (3h):** introduce SDPA behind parity tests, profile prefill/decode, and document exactly what changed.

## Experiments and deliverables

Commit decoder.py, config schema, tensor-contract table, parameter/FLOP calculation, reference-versus-SDPA parity tests, causal-leakage test, gradient test, and a tiny training/generation report. Record B/T/D/H/F/V, dtype/device, tolerance, parameter count, prefill/decode latency, and peak memory.

## Common failure modes

- Mixing [B,T,D] and [B,H,T,Dh] axes or masking the wrong score axis.
- Forgetting scale by sqrt(Dh) or applying causal mask after softmax.
- Treating padding labels as real targets.
- Using SDPA dropout without making its training policy explicit.
- Accidentally untieing embeddings or registering layers incorrectly.
- Comparing implementations with different norm/position/bias/topology choices.

## Mastery gate

From a config, derive all Transformer tensors and parameter counts, then implement a decoder whose reference and SDPA paths pass causal, gradient, state, and randomized parity tests. Explain why an earlier token’s logits cannot depend on a later token and demonstrate the test.

## Primary sources and canonical code paths

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [PyTorch Modules note](https://docs.pytorch.org/docs/stable/notes/modules.html)
- Inspect [functional.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py), [activation.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py), and SDPA/attention tests.

## OSS bridge and next

This reference lets you reduce an upstream attention or model bug to a small parity test. Good contributions often begin with a mask, shape, dtype, or state regression rather than a new kernel. Module 21 adds cached autoregressive decoding, GQA/MQA mappings, and the bridge from this model to real inference engines.
