# Module 45 — Distributed Inference Parallelism

**Phase:** 7 — Distributed runtimes and open source  
**Pacing:** Week 23, paired with Module 46; three 3-hour sessions and a multi-rank communication report  
**Prerequisites:** Modules 05, 12–14, 34, and 44; transformer shapes, KV memory, MoE routing, scheduling, profiling, and process isolation

## Why this matters

More GPUs add memory and compute but also synchronization, collective latency, topology constraints, duplicated state, and new failure modes. Tensor parallelism may make a model fit yet slow batch-one decode because every layer communicates a small message. Expert parallelism can move fewer weights per rank while sending tokens through an imbalanced all-to-all. The correct parallel plan follows model shapes, workload, topology, and SLOs—not GPU count alone.

## Outcomes

You can:

- derive parameter, activation, KV-cache, and communication shapes for TP, PP, DP, EP, and context/sequence parallelism;
- estimate collective time with latency/bandwidth models and validate it using NCCL traces;
- identify which state is sharded, replicated, or routed under each strategy;
- reason about overlap, pipeline bubbles, stragglers, and deadlocks;
- choose and defend a parallel plan for prefill, decode, and MoE workloads.

## Mental model: partition work, then pay to restore dependencies

For \(Y=XW^\top\), with \(X[M,K]\) and \(W[N,K]\):

- **Column parallel:** shard \(W\) along \(N\). Rank \(r\) computes \(Y_r[M,N/P]\). The next operator may consume shards, or an all-gather restores \(Y\).
- **Row parallel:** shard \(K\). Rank \(r\) computes a partial \(Y_r[M,N]\), followed by an all-reduce. Inputs must already be sharded along \(K\).

Alternating column- and row-parallel linears avoids gathering between them, but leaves collective points at residual boundaries. For a ring all-reduce of \(n\) bytes across \(P\) ranks, a useful first model is

\[
T_{\mathrm{AR}}\approx 2(P-1)\alpha+
2\frac{P-1}{P}n\beta,
\]

where \(\alpha\) is per-step latency and \(\beta\) is seconds/byte. Real topology, algorithms, protocols, contention, and reductions change the result. Small decode messages are often latency-dominated; large prefill collectives approach bandwidth limits.

**Pipeline parallelism (PP)** assigns layer ranges to stages and sends activations. With \(m\) microbatches and \(p\) balanced stages, a simple feed-forward utilization ceiling is \(m/(m+p-1)\). Online decode, uneven layers, and communication add bubbles. **Data parallelism (DP)** replicates the model and routes independent requests; it scales service capacity but not single-request model fit. **Expert parallelism (EP)** shards experts and all-to-all routes token representations; capacity factors, hot experts, and padding determine imbalance. **Context/sequence parallelism** shards tokens or attention state, requiring collectives or ring exchanges to recover global attention.

KV cache follows sequences, layers, KV heads, and positions. TP may shard KV heads when divisible; PP shards cache by layer; DP owns separate request caches. Prefix-aware routing and disaggregated prefill/decode add placement and KV-transfer costs that are distinct from model collectives.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/parallel_cost.py` to calculate per-rank weights, KV, activations, collective bytes, and \(\alpha\)-\(\beta\) time for a model/config/workload. Validate tensor-shard reconstruction with small CPU process-group examples.

### 2. Optimized

Implement a two-rank column/row-parallel MLP in `inference-lab/engine/parallel_mlp.py`, with explicit process groups and correctness against an unsharded oracle. If only one GPU is locally available, use CPU/Gloo for semantics and an approved multi-GPU environment for NCCL performance; do not simulate performance claims.

### 3. Integrated

Launch a small engine model under TP or PP, collect per-rank memory and NCCL traces, and sweep prefill/decode shapes. Add one controlled delay or shard mismatch to study straggler and failure behavior. Compare observed communication with the cost model.

## Three 3-hour sessions

1. **Sharding algebra:** derive weight/activation/KV shapes and implement distributed MLP parity.
2. **Collectives and topology:** run `nccl-tests` across message sizes, inspect topology, and fit latency/bandwidth regions.
3. **Engine plan:** trace rank initialization through model execution, profile overlap and bubbles, and recommend a parallel configuration.

## Experiments and deliverables

Commit:

- `reference/parallel_cost.py`;
- `engine/parallel_mlp.py`;
- `tests/test_parallel_shards.py`;
- `benchmarks/results/collective-size-sweep.json`;
- `benchmarks/reports/distributed-plan.md`;
- `notes/source_walks/vllm-distributed.md`.

Record hosts/GPUs, rank mapping, PCIe/NVLink/network topology, process launcher, NCCL/CUDA/runtime versions, environment, tensor shapes, collective algorithms, warm-up, per-rank timing/memory, trace synchronization, and correctness. Never report one rank's time as global latency without confirming the critical rank.

## Common failure modes

- Dividing total parameters by ranks and ignoring replicated embeddings, norms, buffers, and workspaces.
- Confusing DP capacity scaling with single-request acceleration.
- Treating all-reduce bandwidth as raw link bandwidth.
- Benchmarking only large collectives when decode messages are small.
- Oversubscribing a PCIe root or choosing ranks without topology evidence.
- Assuming communication overlaps because it uses another stream.
- Sharding query heads but mishandling GQA KV-head divisibility.
- Letting one rank skip or reorder a collective, causing deadlock.
- Reporting averages that hide the straggling critical rank.
- Applying datacenter NVLink results to a consumer single-GPU workstation.

## Mastery gate

Given model dimensions, \(P\), sequence lengths, and topology, derive per-rank weights/KV plus every MLP/attention communication tensor for a chosen plan. Reconstruct exact outputs in a multi-process test, fit an \(\alpha\)-\(\beta\) model from collective measurements, and explain observed prefill/decode scaling. Diagnose an injected collective-order or shard-shape failure without guessing.

## Canonical code areas and OSS bridge

In vLLM, trace parallel configuration into `vllm/distributed/`, executor/worker rank creation, tensor-parallel layers, pipeline groups, and MoE expert-parallel paths. Compare PyTorch `DeviceMesh`/DTensor and [NCCL](https://github.com/NVIDIA/nccl) primitives. Useful contributions include shard-shape assertions, rank-aware diagnostics, small-message benchmarks, topology documentation, and deadlock reproducers with all rank logs.

## Primary and authoritative sources

- [Megatron-LM paper](https://arxiv.org/abs/1909.08053)
- [GPipe paper](https://arxiv.org/abs/1811.06965)
- [PyTorch tensor-parallel APIs](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NVIDIA `nccl-tests`](https://github.com/NVIDIA/nccl-tests)
- [vLLM parallelism guide](https://docs.vllm.ai/en/latest/configuration/optimization/#parallelism-strategies)

## Next

Module 46 applies the complete curriculum to a vertical vLLM source walk and turns one evidence-backed observation into a maintainable contribution proposal.
