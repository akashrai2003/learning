# Module 18 — Quantization Fundamentals

**Phase:** 4 — Quantization  
**Pacing:** Week 9, paired with Module 17; three 3-hour sessions plus a shared weekend lab  
**Prerequisites:** Modules 03–05 and 17: tensor layouts, floating-point error, roofline reasoning, and architecture manifests

## Why this matters

“Four-bit model” does not specify a computation. Engineers must know which tensors are quantized, the representable set, scale granularity, zero-point convention, storage layout, accumulation type, and available kernel. Quantization can shrink a checkpoint yet slow inference when metadata, unpacking, conversions, or unsupported shapes dominate.

## Outcomes

You will be able to:

- derive symmetric and affine quantizers and their error bounds;
- implement per-tensor, per-channel, and grouped quantization;
- calculate real bits per weight including metadata and padding;
- pack and unpack sub-byte integers without signedness errors;
- connect a quantization recipe to GEMM shapes and kernel constraints.

## Mental model: representation, not compression

An affine integer quantizer maps a real value \(x\) to integer \(q\):

\[
q=\operatorname{clamp}\left(\operatorname{round}(x/s)+z,\ q_{\min},q_{\max}\right),
\qquad
\hat{x}=s(q-z).
\]

Here \(s>0\) is a scale and \(z\) is the integer zero point. For asymmetric min/max calibration,

\[
s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\qquad
z=\operatorname{clamp}\left(\operatorname{round}\left(q_{\min}-x_{\min}/s\right)\right).
\]

For signed symmetric \(b\)-bit quantization, a common choice is \(z=0\),
\(Q=2^{b-1}-1\), and \(s=\max|x|/Q\), using \([-Q,Q]\). Avoid assuming whether the extra two’s-complement value is used; kernels and formats differ.

Without clipping and for round-to-nearest, scalar error satisfies roughly
\(|x-\hat{x}|\le s/2\). Clipping violates that bound, and accumulation can amplify correlated error.

## Granularity and shapes

Let a linear weight be \(W\in\mathbb{R}^{N\times K}\).

- **Per-tensor:** one scale/zero point for all \(NK\) values.
- **Per-output-channel:** \(s,z\in\mathbb{R}^{N}\), broadcast along \(K\).
- **Grouped:** partition each row along \(K\) into groups of \(G\), giving
  \(s,z\in\mathbb{R}^{N\times\lceil K/G\rceil}\).

Smaller groups usually reduce error but increase metadata and complicate kernels. With INT4 weights, group size 128, FP16 scale, and no zero point, ideal storage per full group is

\[
128\cdot4+16=528\text{ bits}=4.125\text{ bits/weight}.
\]

Add zero points, padding, alignment, permutation metadata, and unquantized layers before claiming a model-wide compression ratio.

For \(b=4\), two unsigned codes fit in one byte:

\[
p=(q_0\ \&\ 15)\;|\;((q_1\ \&\ 15)\ll4).
\]

Unpacking uses masks and shifts. Signed INT4 requires an explicit decoding rule such as sign extension or an offset zero point. “Low nibble first” versus “high nibble first,” K-major versus N-major, and pre-shuffled tensor-core layouts are part of the format contract.

## Quantized linear algebra

For activations \(A\in\mathbb{R}^{M\times K}\) and weights
\(W\in\mathbb{R}^{N\times K}\), an integer path forms

\[
C_{mn}=\sum_{k=1}^{K}(q^A_{mk}-z^A)(q^W_{nk}-z^W),
\qquad
Y_{mn}\approx s^A_m s^W_n C_{mn}.
\]

Real kernels may fold zero-point correction into row/column sums, use higher-precision accumulation, and fuse bias or activation in the epilogue. Weight-only kernels instead load FP16/BF16 activations and dequantize packed weights in registers or shared memory while multiplying.

## Implementation ladder

### 1. Reference

Create `inference-lab/reference/quantize.py` with explicit `choose_qparams`, `quantize`, `dequantize`, `pack_int4`, and `unpack_int4`. Make granularity and axis named arguments. Use float64 in the oracle when isolating quantizer error.

### 2. Optimized

Vectorize group reduction and packing; eliminate expanded dequantized weights. Benchmark group sizes 32, 64, 128, and 256 over both matrix-vector decode (\(M\) small) and matrix-matrix prefill (\(M\) large).

### 3. Integrated

Define a serialized tensor contract containing bit width, mapping type, axis/group size, scale dtype, zero-point domain, original shape, padding, and pack order. Dispatch only when the target kernel supports the exact contract; otherwise emit a visible fallback.

## Three 3-hour sessions

1. **Derive:** compute qparams by hand for a vector containing zero, outliers, and a constant range. Derive metadata overhead and INT32 accumulator bounds.
2. **Implement:** write the scalar oracle, vectorized quantizer, and bit-exact packing tests. Test odd lengths and non-divisible groups.
3. **Measure and inspect:** benchmark quantize/dequantize and linear paths; inspect an official sub-byte type and one production quantized tensor layout.

The weekend deliverable is a quantization matrix comparing bit width, granularity, SQNR, maximum error, bytes, and latency for at least four tensor distributions.

## Experiments and deliverables

Add:

- `reference/quantize.py`;
- `tests/test_quantize_properties.py` with round-trip and saturation properties;
- `benchmarks/results/quantizer-granularity.json`;
- `notes/quantization/fundamentals.md`;
- a format manifest under `benchmarks/manifests/`.

Use fixed random seeds and include Gaussian, heavy-tailed, outlier-channel, and constant tensors. Report quantizer time separately from GEMM time and compare against a BF16/FP32 oracle.

## Common failure modes

- Calling packed storage a dtype while omitting scale semantics.
- Mixing weight convention \([N,K]\) with a kernel expecting \([K,N]\).
- Computing a zero scale for constant tensors.
- Relying on host-language signed right shifts.
- Ignoring group tails, padding, endianness, or nibble order.
- Reporting `numel * bits` as file or GPU memory.
- Comparing only tensor MSE; small structured errors may change logits materially.

## Mastery gate

From a written format contract, implement bit-exact quantize/dequantize and packing, derive effective bits per weight, bound accumulator magnitude, and explain whether the corresponding decode GEMM is likely compute- or bandwidth-limited.

## Canonical code paths and OSS bridge

Study PyTorch’s [quantization API equations and granularity types](https://docs.pytorch.org/docs/stable/quantization-support), TorchAO’s [`Int4WeightOnlyConfig`](https://github.com/pytorch/ao), and CUTLASS [sub-byte fundamental types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html). Trace from a serialized quantization config to the runtime’s parameter loader and then its kernel pack/dispatch code. Useful OSS work includes property tests for tail groups, clearer format validation, or documenting a layout that currently exists only in kernel code.

## Primary and authoritative sources

- [Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [PyTorch affine quantization support](https://docs.pytorch.org/docs/stable/quantization-support)
- [PyTorch MinMaxObserver](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver.html)
- [CUTLASS fundamental types](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/fundamental_types.html)

## Next

Module 19 asks how qparams should be calibrated and how numerical error becomes—or fails to become—model-quality degradation.
