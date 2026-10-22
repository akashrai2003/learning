# Module 21 — Weight-Only INT8/INT4 and Packing

**Phase:** 4 — Quantization  
**Pacing:** Week 11, paired with Module 22; three 3-hour sessions plus a weekend kernel/layout report  
**Prerequisites:** Modules 05 and 18–20: roofline analysis, grouped affine quantization, packing, and evaluation

## Why this matters

Autoregressive decode repeatedly streams model weights for a small number of tokens. Weight-only quantization attacks that bandwidth cost while leaving activations and accumulation in FP16/BF16 or FP32. But an INT4 file is not automatically a fast W4A16 operator: dequantization, scale loads, padding, reshuffling, and kernel support determine the result.

## Outcomes

You can:

- design per-channel and grouped INT8/INT4 weight formats;
- calculate packed size, metadata overhead, and alignment waste;
- explain fused dequantize-and-multiply kernels;
- distinguish decode-friendly matrix-vector shapes from prefill GEMM;
- validate format compatibility from checkpoint through kernel dispatch.

## Mental model: trade weight bytes for decode work

Let \(X\in\mathbb{R}^{M\times K}\), stored weights
\(W\in\mathbb{R}^{N\times K}\), and \(Y=XW^\top\). Weight-only quantization stores codes \(Q\), scales \(s\), and possibly zero points \(z\), but \(X\) remains BF16/FP16:

\[
\hat W_{n,k}=s_{n,g(k)}(Q_{n,k}-z_{n,g(k)}),
\qquad
\hat Y_{m,n}=\sum_kX_{m,k}\hat W_{n,k}.
\]

With group size \(G\), \(s,z\) have shape
\([N,\lceil K/G\rceil]\). An efficient kernel loads packed codes and group metadata, reconstructs fragments in registers, and immediately feeds multiply-accumulate instructions; materializing \(\hat W\) in HBM loses the bandwidth benefit.

For symmetric INT4, two 4-bit codes occupy one byte. For asymmetric groups, unsigned codes \(0\ldots15\) pair with a zero point. Effective storage is

\[
b_\text{eff}=b+
\frac{b_s+b_z}{G}+
\frac{\text{padding+alignment bits}}{NK}.
\]

Example: INT4, \(G=128\), FP16 scales, and packed 4-bit zero points costs at least \(4+16/128+4/128=4.15625\) bits/weight before alignment and unquantized tensors.

## Packing is a kernel ABI

A logical row-major \([N,K]\) nibble array is only the first representation. Tensor-core kernels often require tiles interleaved across K and N, lane-specific permutations, or pre-shuffled scales. Record:

- logical and physical shapes;
- group axis and group size;
- nibble order and signed encoding;
- padding of \(K\), \(N\), and groups;
- scale/zero-point dtype and layout;
- any column permutation and how it is inverted;
- architecture/version identifier for the packer.

Never infer compatibility because two formats are called “GPTQ INT4.” A packer and consumer must share the exact contract.

## Performance model

Decode often has \(M=1\) or a small active batch, so arithmetic intensity is low and reducing weight bytes can matter greatly. Prefill has large \(M\), reuses each weight across many rows, and may favor a native BF16/FP8 GEMM over W4A16 dequantization. Measure both regimes.

Estimate bytes per output tile: packed weights + scales + zero points + activations + outputs. Then add unpack/conversion instructions. A smaller group reduces numerical error but increases metadata bandwidth and scale changes inside the K loop. Shapes not divisible by kernel tile sizes may use a slower tail or fall back.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/weight_only.py`. Quantize \(W[N,K]\) groupwise, pack/unpack bit-exactly, dequantize to FP32, and compare `X @ W_hat.T` to the original. Keep the serialized format dataclass explicit.

### 2. Optimized

Write a vectorized or Triton prototype that unpacks and dequantizes inside the matrix multiply rather than allocating a full temporary. Specialize separate \(M=1\) and \(M\gg1\) paths. Benchmark INT8 and INT4 over \(G\in\{32,64,128\}\).

### 3. Integrated

Export a small model with pinned format metadata, load it in one engine, and capture the chosen kernel and any repacking at startup. Compare cold load cost, steady-state decode, prefill, GPU memory, and quality. A visible reference fallback is acceptable; a silent one invalidates a speed claim.

## Three 3-hour sessions

1. **Format design:** derive effective bits/weight and implement signed/asymmetric INT4 pack/unpack with odd dimensions.
2. **Operator work:** implement naive dequantize-then-GEMM and fused/on-the-fly variants; use a profiler to account for temporary bytes.
3. **Production trace:** follow checkpoint metadata through loader, parameter transform, kernel selection, and launch for one runtime.

The weekend report should include a roofline hypothesis and a shape sweep separating \(M=1,8,128,1024\).

## Experiments and deliverables

Commit:

- `reference/weight_only.py`;
- `tests/test_weight_pack.py` with golden byte strings;
- `kernels/triton/weight_only_linear.py` or a documented pseudokernel;
- `benchmarks/results/weight-only-shapes.json`;
- `notes/source_walks/weight-pack-to-kernel.md`.

Test all-zero, constant, odd-K, outlier-group, transposed, and noncontiguous inputs. Quality evaluation must include PPL/logit metrics from Module 19, not only weight MSE.

## Common failure modes

- Timing GEMM after pre-dequantizing weights outside the measured region.
- Confusing mathematical INT4 with packed physical layout.
- Ignoring scale/zero-point and padding overhead.
- Comparing a warmed quantized kernel with a cold BF16 path.
- Using one \(M\) and generalizing to both prefill and decode.
- Saving a column permutation without restoring activation order.
- Quantizing embeddings or LM head without evaluating their sensitivity.

## Mastery gate

Given \(N,K,G\), a pack specification, and target tile constraints, calculate exact storage, implement a golden pack/unpack test, predict decode versus prefill behavior, and prove with profiler evidence that no full dequantized weight is written to HBM.

## Canonical code paths and OSS bridge

Study TorchAO’s [INT4 weight-only workflow and layouts](https://github.com/pytorch/ao), CUTLASS [packed sub-byte types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html), and TensorRT-LLM’s [weight-only format options](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md). In vLLM, trace quantization configuration into linear-kernel selection and parameter repacking. Good OSS contributions are golden layout tests, unsupported-shape diagnostics, or decode-shaped benchmark cases.

## Primary and authoritative sources

- [TorchAO repository](https://github.com/pytorch/ao)
- [CUTLASS efficient GEMM design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [CUTLASS fundamental types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html)
- [TensorRT-LLM quantization examples](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md)

## Next

Module 22 replaces independent rounding with activation-aware, approximate second-order error compensation: GPTQ.
