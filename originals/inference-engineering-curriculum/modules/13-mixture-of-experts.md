# Module 13 — Mixture of Experts

> **Phase:** 3 — Advanced Architectures  
> **Time:** Week 7; two 3-hour sessions plus a 2–4 hour independent lab and mastery gate  
> **Prerequisites:** Modules 03–12, especially SwiGLU, tensor layouts, quantization error awareness, and prefill/decode workload differences

## Why this matters

Sparse mixture-of-experts (MoE) models increase parameter capacity without activating every parameter for every token. That does not make them small: all expert weights must be stored somewhere, and inference adds routing, permutation, grouped GEMMs, imbalance, and often all-to-all communication. A naive Python implementation can make a theoretically sparse layer slower than a dense FFN. Understanding the full route–dispatch–compute–combine pipeline is essential for serving Mixtral-like and newer MoE checkpoints.

## Outcomes

You will be able to:

- derive router, expert, and output shapes for top-$k$ token routing;
- distinguish total, resident, and per-token active parameters;
- implement a correct sparse MoE reference with checkpoint-specific weighting;
- explain token sorting, grouped GEMM, expert parallelism, and load imbalance; and
- measure routing distributions rather than attributing every slowdown to GEMM.

## Mental model

An MoE layer is a switchboard in front of a bank of FFNs. For every token, a router selects a small set of experts and supplies mixing weights. The engine copies—or logically indexes—tokens into expert-specific queues, runs each queue through its FFN, then scatters and sums results back into original token order. Sparsity saves expert computation, but the switchboard and queues are real work. During decode, a small batch may give each selected expert only a few rows, producing inefficient tiny matrix multiplications.

## Theory, math, and shapes

Flatten hidden states from `[B,T,d]` to `X: [M,d]`, where `M=B*T`. A router matrix `W_r: [d,E]` produces logits

$$
R=XW_r\in\mathbb{R}^{M\times E}.
$$

For token $i$, choose top-$k$ expert indices $e_{ij}$ and weights $a_{ij}$, then compute

$$
y_i=\sum_{j=1}^{k} a_{ij}\,\operatorname{Expert}_{e_{ij}}(x_i).
$$

`topk_ids` and `topk_weights` are `[M,k]`; each expert input is `[M_e,d]`, and the final output returns to `[M,d]`. An expert is commonly the SwiGLU FFN from Module 09. Stacked weights may have shapes such as `[E,2f,d]` for packed gate/up and `[E,d,f]` for down projection.

Router semantics are checkpoint-specific. Some implementations softmax across all `E` logits and then select; some select logits and renormalize selected weights; some apply corrections or shared experts. Preserve the exact order. Top-1 Switch routing and top-2 Mixtral routing are not interchangeable.

If every expert has $P_e$ parameters, total expert parameters are $EP_e$, but a token activates roughly $kP_e$. All $EP_e$ remain resident across one device or an expert-parallel group. With expert parallelism, tokens cross devices according to routing, so imbalance and communication may dominate. Capacity factors and auxiliary load-balancing losses mainly concern training; inference must not silently drop overflow tokens unless the architecture explicitly defines that behavior.

## Implementation ladder

1. Create `inference-lab/reference/moe.py` with a deterministic router and a `ModuleList` of small SwiGLU experts. Implement a slow token loop as the oracle.
2. Add a packed implementation: flatten token/expert assignments, sort by expert, gather token rows, compute expert batches, multiply routing weights, and scatter-add by original token ID.
3. Store expert weights in a stacked layout in `kernels/grouped_moe.py`; use ordinary batched operations before attempting a custom grouped-GEMM kernel.
4. Add `MoEDecoderBlock` to `reference/decoder_block.py` behind configuration. Preserve residual and normalization order.
5. Instrument counts, fraction of tokens per expert, empty experts, maximum/mean load, dispatch bytes, and route/compute/combine timings.

## Two 3-hour sessions

**Session A — routing correctness**

- 0:00–0:40: derive top-1 and top-2 routing on a hand-worked batch.
- 0:40–1:20: implement the loop oracle and checkpoint-selectable normalization.
- 1:20–2:20: implement sort, gather, expert compute, and scatter-add.
- 2:20–3:00: test order restoration, empty experts, ties, and skewed routes.

**Session B — systems behavior**

- 0:00–0:45: stack expert weights and identify grouped-GEMM dimensions.
- 0:45–1:30: integrate the MoE block and add stage-level timers.
- 1:30–2:20: benchmark prefill and decode batches under balanced and skewed routing.
- 2:20–3:00: model expert-parallel communication and compare it with measured bytes.

## Experiments and deliverables

Commit:

- `tests/test_moe.py` proving loop-versus-packed parity for `k={1,2}`, empty experts, repeated token assignments, and deterministic ties;
- `benchmarks/bench_moe.py` sweeping token count, expert count, `k`, width, and routing skew;
- `notes/module13.md` with total/resident/active parameter arithmetic and a route histogram;
- a breakdown of router, permutation, first GEMM, activation, second GEMM, and combine time; and
- one adversarial case where every token selects the same expert, with an explanation of its utilization.

Compare a parameter-matched dense FFN and an MoE by both latency and bytes of resident weights. Do not report “active parameters” as though they describe VRAM consumption.

## Common failure modes

- Softmax-before-top-$k$ versus top-$k$-then-softmax mismatch.
- Forgetting to multiply or renormalize routing weights.
- Losing original token order during dispatch or using overwrite instead of scatter-add for `k>1`.
- Dropping tokens because a training-time capacity rule leaked into inference.
- Assuming balanced random routing represents production prompts.
- Launching one Python loop and one tiny kernel per expert.
- Quantizing router logits so aggressively that top-$k$ identities change.
- Loading all expert weights on every tensor-parallel rank when expert parallelism was intended.
- Measuring only expert GEMM while ignoring permutation and collectives.
- Confusing a shared expert, which runs for every token, with routed experts.

## Mastery gate

Given `M=1024, d=4096, E=8, k=2, f=14336`, derive router and packed-assignment shapes, active expert FFN FLOPs, and total expert parameter storage. Your loop and packed paths must agree for balanced, empty-expert, and maximally skewed cases. You pass when your benchmark explains a crossover where packed sparse execution becomes faster than the loop and why that crossover shifts during decode.

## Primary sources

- Shazeer et al., [“Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”](https://arxiv.org/abs/1701.06538).
- Fedus, Zoph, and Shazeer, [“Switch Transformers”](https://arxiv.org/abs/2101.03961).
- Jiang et al., [“Mixtral of Experts”](https://arxiv.org/abs/2401.04088).
- Gale et al., [MegaBlocks paper](https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf) and [canonical repository](https://github.com/databricks/megablocks).

## Code paths to read

Your paths are `reference/moe.py`, `kernels/grouped_moe.py`, `reference/decoder_block.py`, `tests/test_moe.py`, `benchmarks/bench_moe.py`, and `notes/module13.md`.

## OSS bridge

Trace Transformers’ [`MixtralSparseMoeBlock`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py). Then inspect vLLM’s [`fused_moe.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py), especially sorted token IDs, expert IDs, block alignment, and routed-weight multiplication. Relate those fields to your packed reference before reading kernel code.

## Next

Module 14 scales sequence state rather than parameter capacity: long-context attention, paged allocation, prefix reuse, and scheduling.
