# Capstone 01 — Reference Decoder

## Objective

Implement a small decoder-only Transformer inference path that is transparent
enough to serve as the correctness oracle for later quantization, attention,
kernel, and serving work.

This is not a training framework and not a reimplementation of all of
Transformers. The goal is to make every tensor transformation, cache mutation,
and decoding decision inspectable.

## Scope

Required:

- tokenizer adapter with explicit special-token policy;
- token embeddings and optional tied LM head;
- RMSNorm and LayerNorm references;
- RoPE;
- MHA and GQA;
- causal masking;
- SwiGLU/gated FFN;
- prefill and one-token cached decode;
- greedy, temperature, top-k, and top-p decoding;
- checkpoint-tensor mapping for one small open model;
- CPU and CUDA/PyTorch execution.

Non-goals:

- training or autograd performance;
- production API serving;
- every tokenizer/model family;
- hand-written GPU kernels;
- continuous batching.

## Milestones

### A. Shape-safe block

Write one decoder block with assertions documenting:

```text
tokens         [B, S]
hidden         [B, S, D]
q              [B, Hq, S, Dh]
k, v           [B, Hkv, S, Dh]
scores         [B, Hq, S, T]
ffn gate/up     [B, S, F]
logits          [B, S, V]
```

Test MHA first, then GQA broadcasting. Do not use a production fused attention
operator as the oracle.

### B. Full-sequence parity

Load a tiny compatible checkpoint and compare:

- each mapped parameter;
- embedding output;
- one block output;
- final hidden state;
- logits for fixed inputs;
- greedy generated tokens.

Record tolerances by dtype and explain differences caused by operation order.

### C. KV-cached decoding

Implement cache allocation and append/update semantics. For every position
\(t\), compare cached token logits to a full forward over tokens \(0\ldots t\).
Include multi-token prefill followed by decode.

### D. Decoder library

Separate:

- model semantics;
- checkpoint loading;
- cache storage;
- sampling;
- benchmark harness.

Later projects must be able to substitute a quantized linear, attention
implementation, or paged cache without rewriting the model.

## Correctness matrix

At minimum test:

- batch sizes 1 and 2;
- prompt lengths 1, 2, and a non-power-of-two length;
- MHA and GQA head configurations;
- FP32 oracle and BF16/FP16 execution where supported;
- left/right padding only if the chosen API claims support;
- empty/invalid inputs rejected clearly;
- stop token, maximum token, and all-masked logits behavior;
- deterministic repeatability for greedy decoding.

## Performance study

Do not optimize yet. Instrument:

- time and bytes by embeddings, attention, FFN, normalization, and LM head;
- prefill versus decode;
- model weights, activations, and cache memory;
- theoretical FLOPs and arithmetic intensity;
- PyTorch eager versus `torch.compile` as an observation, not a requirement.

Explain why the dominant operation changes between prefill and single-token
decode.

## Review package

Deliver:

- `reference/` implementation;
- unit and end-to-end tests;
- checkpoint mapping note;
- shape/FLOP/byte table;
- one source walk through the same architecture in a production runtime;
- benchmark manifest and raw results;
- five-minute whiteboard explanation recorded or rehearsed.

## Acceptance gate

Starting from a blank page, derive the decoder block shapes, write cached
attention pseudocode, and diagnose an intentionally introduced cache-position
or RoPE offset bug. If cached and full logits disagree, you must localize the
first layer and tensor where divergence appears.

## Stretch work

- beam search with explicit cache reordering;
- sliding-window attention;
- multimodal embedding insertion;
- MoE block;
- safetensors zero-copy/memory-mapped loader;
- ONNX or another graph export, with a report on dynamic-cache limitations.

