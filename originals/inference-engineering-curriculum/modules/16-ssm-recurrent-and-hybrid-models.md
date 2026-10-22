# Module 16 — SSM, Recurrent, and Hybrid Models

> **Phase:** 3 — Advanced Architectures  
> **Time:** Week 8; two 3-hour sessions plus a 2–4 hour integration lab and mastery gate  
> **Prerequisites:** Modules 03–15; causal convolutions, scan/reduction intuition, stateful decoding, paging, batching, and numerical precision

## Why this matters

Transformers store a growing K/V history and reread it during decode. Recurrent and state-space models summarize history into fixed-shape state, offering sequence-length-independent decode-state memory for pure recurrent layers. Mamba makes state updates input-dependent and supplies hardware-aware parallel scans for prefill. Production models may interleave Mamba, attention, sliding-window, and MoE layers, forcing an engine to manage several state types and rollback rules in one request.

## Outcomes

You will be able to:

- derive a discrete state-space recurrence and its tensor state;
- explain selective state updates and parallel-scan versus recurrent execution;
- implement matching full-sequence and one-step reference paths;
- compare Transformer KV growth with recurrent-state memory; and
- design lifecycle, batching, preemption, and snapshot rules for hybrid models.

## Mental model

Attention is an archive: it retains records for past tokens and searches them for every query. A recurrent SSM is a running ledger: each token updates a fixed-size summary, and the original records need not remain. The ledger is compact in sequence length but lossy and architecture-specific. Hybrid models keep both—occasional attention archives for precise retrieval and recurrent ledgers for efficient continuous processing.

## Theory, math, and shapes

A continuous linear SSM has $\dot h(t)=Ah(t)+Bx(t)$ and $y(t)=Ch(t)+Dx(t)$. After discretization, an inference recurrence is

$$
h_t=\bar A_t\odot h_{t-1}+\bar B_t\odot x_t,\qquad
y_t=C_t h_t+D x_t .
$$

The diagonal/structured form shown is simplified but useful. Classical SSM parameters are fixed. Mamba's selectivity makes step size and input/output terms functions of the current token, allowing the model to retain or forget content conditionally.

For a typical Mamba-style block, input is `[B,T,d_model]`, an expansion produces `[B,T,d_inner]`, a short depthwise causal convolution captures local context, and the SSM carries state roughly `[B,d_inner,d_state]`. Decode also retains convolution history around `[B,d_inner,d_conv-1]`. Exact layouts differ across Mamba versions and checkpoints.

Sequential recurrence costs $O(T)$ steps and is ideal for decode. Prefill would underutilize a GPU if executed as a Python loop. Selective scan reformulates compatible updates as an associative scan, processing a sequence in parallel while avoiding materializing all expanded state in HBM. Mamba-2's state-space duality connects structured SSM computation with attention-like matrix structure and improves kernel efficiency.

For the affine update $h'=a\odot h+b$, summarize a segment by `(a,b)`. Two consecutive segments compose as `(a2*a1, a2*b1+b2)` elementwise. This operator is associative, which is the algebraic reason a prefix scan can reproduce a sequential recurrence in parallel; selectivity changes how the per-token pairs are produced, not that composition law.

A pure recurrent layer's live decode state is constant with context length, but not necessarily small: multiply both state tensors by layers, sequences, and bytes per element. A hybrid model still has growing KV for attention layers. Prefix caching may store recurrent snapshots at block boundaries; speculative decoding or preemption may require copy/restore rather than merely shortening a length counter.

## Implementation ladder

1. Add `inference-lab/reference/ssm.py` with a scalar exponential moving recurrence, then a diagonal vector SSM. Provide `forward_sequence(x,h0)` and `step(x_t,h)`.
2. Add input-dependent gates and a depthwise causal-convolution state. Keep all reference state updates out-of-place.
3. Implement an associative scan in `kernels/scan.py` for the non-selective affine recurrence; use it to understand composition before studying optimized selective kernels.
4. Add `engine/recurrent_state.py` with allocate, reset, clone, reorder, snapshot, restore, and per-request ownership.
5. Build `reference/hybrid_block.py` that follows configuration to alternate attention and SSM layers. Extend scheduling/cache metadata to carry both KV block tables and recurrent state.

## Two 3-hour sessions

**Session A — recurrence and scan**

- 0:00–0:45: discretize a simple SSM and unroll four steps by hand.
- 0:45–1:30: implement scalar, vector, and input-selective recurrent steps.
- 1:30–2:15: add convolution state and full-versus-step equivalence tests.
- 2:15–3:00: derive affine-update composition and implement a parallel scan.

**Session B — inference lifecycle**

- 0:00–0:40: calculate recurrent, KV, and hybrid state footprints.
- 0:40–1:30: implement batch reorder, snapshot/restore, and request isolation.
- 1:30–2:15: assemble and test a configurable hybrid block.
- 2:15–3:00: benchmark prefill and decode scaling; inspect a production kernel path.

## Experiments and deliverables

Produce:

- `tests/test_ssm.py` proving full-sequence, irregular-chunk, and one-step parity;
- `tests/test_recurrent_state.py` covering reset, reorder, snapshot/restore, and two-request isolation;
- `benchmarks/bench_ssm.py` comparing a sequential reference, scan path, and attention baseline across length;
- a memory plot showing recurrent state, full-attention KV, and a chosen hybrid as context grows;
- a failed-speculation experiment that restores state and reproduces baseline logits; and
- `notes/module16.md` mapping mathematical state to concrete tensor layout and dtype.

Measure prefill and decode separately. A fast recurrent decode does not imply an efficient unfused prefill, and a Python scan result says little about a hardware-aware fused kernel.

## Common failure modes

- Reusing one mutable state tensor across requests or beams.
- Updating state twice when a sampled/speculative token is rejected or replayed.
- Resetting SSM state but not causal-convolution history.
- Treating padding tokens as real recurrence steps in a ragged batch.
- Assuming recurrent state can be “cropped” like KV cache; rollback generally needs a snapshot or recomputation.
- Accumulating long recurrences in an unstable low-precision dtype without model guidance.
- Materializing `[B,T,d_inner,d_state]` during prefill and losing the scan's memory benefit.
- Applying Mamba-1 state shapes to Mamba-2 or another recurrent architecture.
- Claiming constant memory for a hybrid while ignoring its attention layers.
- Expecting generic Transformer kernels, prefix caching, or speculative decoding to work without model-specific state hooks.

## Mastery gate

Derive and implement composition of two affine updates `(A2,B2) ∘ (A1,B1)` and use it in a scan. Sequence, chunked, and step paths must agree for at least five lengths. Demonstrate correct snapshot/restore after rejected tokens and derive the per-request state bytes for a supplied hybrid configuration. You pass when no state leaks across interleaved requests and your benchmark explains the different prefill/decode scaling.

## Primary sources

- Gu and Dao, [“Mamba: Linear-Time Sequence Modeling with Selective State Spaces”](https://arxiv.org/abs/2312.00752).
- Dao and Gu, [“Transformers are SSMs” (Mamba-2 / state-space duality)](https://arxiv.org/abs/2405.21060).
- Peng et al., [“RWKV: Reinventing RNNs for the Transformer Era”](https://arxiv.org/abs/2305.13048).
- Lieber et al., [“Jamba: A Hybrid Transformer-Mamba Language Model”](https://arxiv.org/abs/2403.19887).
- State Spaces, [canonical Mamba implementation](https://github.com/state-spaces/mamba).

## Code paths to read

Your paths are `reference/ssm.py`, `kernels/scan.py`, `engine/recurrent_state.py`, `reference/hybrid_block.py`, `tests/test_ssm.py`, `tests/test_recurrent_state.py`, and `benchmarks/bench_ssm.py`.

## OSS bridge

In the Mamba repository, trace [`selective_scan_interface.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py), [`mamba_simple.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py), and the one-step update kernel. Then read vLLM’s [hybrid KV-cache manager design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) and model interface state-shape hooks. Identify where page-size uniformity forces padding between unlike cache groups.

## Next

Module 17 uses these architecture-specific state contracts while reconstructing an unfamiliar model from config, checkpoint tensors, and production code.
