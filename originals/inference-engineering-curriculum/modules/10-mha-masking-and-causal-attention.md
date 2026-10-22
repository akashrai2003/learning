# Module 10 — MHA, Masking, and Causal Attention

> **Phase:** 2 — Architecture Core  
> **Time:** Week 5; two 3-hour sessions plus a 2–4 hour independent lab and mastery gate  
> **Prerequisites:** Modules 03–09; matrix multiplication, stable softmax, residual blocks, and basic GPU memory hierarchy

## Why this matters

Causal self-attention is the semantic core of a Transformer decoder and one of its most failure-prone implementations. A transposed dimension, reversed boolean mask, missing scale, or incorrect cache offset can yield fluent but invalid generation. Efficient inference further requires separating the mathematical operation from its implementation: eager attention, PyTorch SDPA, FlashAttention, and paged-attention kernels should compute the same result while moving very different amounts of data.

## Outcomes

You will be able to:

- derive all projection, score, probability, and output shapes in multi-head attention (MHA);
- construct causal and padding masks for square and non-square attention;
- implement a numerically stable reference and validate an optimized backend;
- explain why FlashAttention is exact yet avoids materializing the full score matrix; and
- distinguish prefill behavior from one-token decode behavior.

## Mental model

Each query is a reader. Keys are addresses describing what stored tokens contain, and values are the payloads returned from those addresses. A dot product ranks addresses; softmax turns ranks into mixing weights. Multiple heads create independent address spaces. The causal mask is an access-control rule: token position `i` may read its past and itself, never a future token. It is part of model semantics, not an optional speed feature.

## Theory, math, and shapes

For hidden states `X: [B,T,d]`, let `H` be the head count and `D=d/H`. Projections produce `Q,K,V: [B,T,H,D]`, normally transposed to `[B,H,T,D]`. With query length `L` and key length `S`,

$$
A=\frac{QK^\top}{\sqrt D}+M,\qquad
P=\operatorname{softmax}(A,\text{dim}=-1),\qquad
O=PV.
$$

`A` and `P` are `[B,H,L,S]`; `O` is `[B,H,L,D]`, then transposed, made contiguous if necessary, reshaped to `[B,L,d]`, and output-projected.

An additive mask contains `0` for allowed edges and `-∞` for forbidden edges. For training or full prefill, causal access is `key_position <= query_position`. With `p` cached tokens and a new query chunk, row `i` may access keys through `p+i`; using a fresh `tril(L,S)` without this offset is wrong. A padding mask removes invalid key positions per sequence and composes with causality. Boolean-mask polarity differs among APIs, so convert it once at the boundary and test the contract.

The scale $1/\sqrt D$ keeps score variance controlled. Compute softmax using a stable maximum subtraction, usually accumulating in float32 even when inputs are fp16/bf16. A naive implementation stores `O(BHLS)` scores and probabilities. FlashAttention tiles Q, K, and V through on-chip memory and maintains online softmax statistics, producing exact attention with far fewer high-bandwidth-memory reads and linear auxiliary memory.

## Implementation ladder

1. Build `inference-lab/reference/masks.py` with explicit additive causal and padding masks. Print tiny `L=3,S=5,p=2` examples before generalizing.
2. Implement `attention_reference` in `reference/attention.py`: float32 scores, explicit mask addition, stable softmax, and output multiplication.
3. Add projection and head reshape logic in `MultiHeadAttention`; assert `d % H == 0` and document the chosen `[B,H,T,D]` layout.
4. Add a backend adapter in `kernels/attention.py` for `torch.nn.functional.scaled_dot_product_attention`. Keep reference and optimized code independent.
5. Integrate attention before the FFN in `reference/decoder_block.py`, preserving normalization and residual order from the target architecture.

Do not add a KV cache yet; Module 12 will introduce its state and non-square decode semantics deliberately.

## Two 3-hour sessions

**Session A — attention semantics**

- 0:00–0:40: derive one-head attention by hand for three tokens.
- 0:40–1:20: construct and visualize causal, padding, and combined masks.
- 1:20–2:20: implement the reference and MHA projections.
- 2:20–3:00: test head splitting, masked probabilities, and output shapes.

**Session B — optimized exact attention**

- 0:00–0:35: derive stable and online softmax recurrences.
- 0:35–1:20: add the SDPA adapter and backend-selection logging.
- 1:20–2:10: test square prefill and offset, non-square chunk cases.
- 2:10–3:00: benchmark sequence-length scaling and interpret memory traces.

## Experiments and deliverables

Produce:

- `tests/test_attention.py` comparing the reference with SDPA over random shapes, dtypes, padding patterns, and causal modes;
- an information-leak test: perturb future values and prove earlier outputs do not change;
- a row-sum check for valid queries and an explicit policy for fully masked rows;
- `benchmarks/bench_attention.py` sweeping `T={64,256,1024,4096}` and recording latency plus peak memory; and
- `notes/module10.md` separating the algorithm's FLOPs, auxiliary memory, and HBM traffic.

For a tiny deterministic tensor, save Q, K, scores before/after masking, probabilities, and output. This trace is the fastest future debugging oracle.

## Common failure modes

- Softmax over the query dimension instead of the key dimension.
- Scaling after softmax, or scaling by `sqrt(d)` rather than `sqrt(D)`.
- A boolean mask whose `True` meaning is reversed at an API boundary.
- Applying only causality and allowing attention to padded keys.
- Building a square causal mask during cached/chunked decoding.
- Calling dropout-enabled SDPA during evaluation; the functional API applies the supplied probability regardless of module mode.
- Viewing a non-contiguous transpose without making layout requirements explicit.
- Filling every entry of a row with `-inf`, producing NaNs in some implementations.
- Treating different low-precision reduction orders as a correctness failure without an error budget.

## Mastery gate

Given `Q: [2,8,3,64]`, cached `K,V` length 11, and three new tokens, state the score shape and draw the exact allowed-key boundary for every query row. Implement that case and pass reference-versus-SDPA checks plus the future-leak test. Finally, explain from an HBM perspective why an exact tiled kernel can outperform naive attention without reducing the $O(T^2D)$ arithmetic of prefill.

## Primary sources

- Vaswani et al., [“Attention Is All You Need”](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf).
- PyTorch, [`scaled_dot_product_attention` API and reference semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
- Dao et al., [“FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”](https://arxiv.org/abs/2205.14135).
- Dao-AILab, [official FlashAttention implementation and tests](https://github.com/Dao-AILab/flash-attention).

## Code paths to read

Your paths are `reference/masks.py`, `reference/attention.py`, `kernels/attention.py`, `reference/decoder_block.py`, `tests/test_attention.py`, and `benchmarks/bench_attention.py`.

## OSS bridge

Compare your eager equation with Transformers’ [`eager_attention_forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py). Then trace vLLM’s [`Attention` layer](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/attention.py) into a selected backend. Record where layout, metadata, and kernel dispatch become separate responsibilities.

## Next

Module 11 adds position through RoPE and studies what context scaling changes—and what it cannot guarantee.
