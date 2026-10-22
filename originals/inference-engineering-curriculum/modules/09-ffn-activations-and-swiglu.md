# Module 09 — FFNs, Activations, and SwiGLU

> **Phase:** 2 — Architecture Core  
> **Time:** Week 5; two 3-hour sessions plus a 2–4 hour independent lab and mastery gate  
> **Prerequisites:** Modules 03–08; comfortable with tensor shapes, GEMM, residual blocks, RMSNorm, and PyTorch broadcasting

## Why this matters

The feed-forward network (FFN) often owns most of a decoder layer's parameters and a large fraction of its FLOPs. It can therefore dominate weight traffic during token-by-token decoding even though attention receives more discussion. Modern decoder models also rarely use the original Transformer's two-matrix ReLU FFN: Llama-family models and many peers use a three-matrix gated FFN, usually SwiGLU. Correct checkpoint loading, quantization, tensor parallelism, and kernel fusion all depend on recognizing that structure.

## Outcomes

By the end, you can:

- derive parameter counts, FLOPs, and intermediate shapes for dense and gated FFNs;
- implement ReLU, GELU, SiLU, GLU, and SwiGLU without accidental broadcasting;
- explain why a parameter-matched SwiGLU uses a smaller intermediate width than a conventional FFN;
- validate a fused or quantized FFN against a high-precision reference; and
- identify when decode performance is limited by reading FFN weights rather than arithmetic.

## Mental model

Treat an FFN as a per-token feature workshop. The first projection expands a token from model width into many candidate features; the nonlinearity decides which features pass; the final projection mixes them back into the residual stream. A gated FFN installs two expansion paths. One proposes content, while the other produces a data-dependent valve. Their elementwise product is compressed by the down projection. Tokens never mix with one another here: flattening batch and time is semantically safe.

## Theory, math, and shapes

Let input `X` have shape `[B, T, d]`, and let `M = B*T`. A conventional FFN is

$$
Y = \phi(XW_{\text{up}})W_{\text{down}},
$$

with `W_up: [d, f]`, activation `[B,T,f]`, and `W_down: [f,d]`. Ignoring bias, it has `2df` parameters and approximately `4Mdf` floating-point operations because a multiply-add counts as two FLOPs.

GELU is $x\Phi(x)$; SiLU (also called Swish) is $x\sigma(x)$. A SwiGLU FFN is

$$
G=XW_g,\quad U=XW_u,\quad
Y=(\operatorname{SiLU}(G)\odot U)W_d .
$$

`G` and `U` are both `[B,T,f]`, so their product must be elementwise. The three weight matrices contain `3df` parameters and cost roughly `6Mdf` FLOPs. To compare fairly with a conventional FFN using `f=4d`, choose a gated width near `f=8d/3`, often rounded for hardware alignment. Real checkpoints specify the exact width; never infer it from a rule of thumb.

PyTorch stores `nn.Linear` weights as `[out_features, in_features]`, the transpose of the mathematical notation above. Bias presence, activation choice, projection names, and whether gate/up weights are packed are checkpoint contracts.

## Implementation ladder

1. In `inference-lab/reference/activations.py`, implement activation formulas in float64 and compare them with PyTorch. Include extreme inputs such as `-100`, `0`, and `100`.
2. In `inference-lab/reference/ffn.py`, implement `DenseFFN` and `SwiGLUFFN` using three explicit linears. Assert every intermediate shape and retain a float32 reference path.
3. Add `inference-lab/kernels/swiglu.py` with a fused activation-times-up operation. Fusion should remove an intermediate write, not change the equation.
4. Wire a selectable FFN into `inference-lab/reference/decoder_block.py`. Load a tiny synthetic state dict by explicit name and shape rather than positional order.
5. Add dtype-aware tolerances and benchmark hooks; only then try compilation or a fused external kernel.

## Two 3-hour sessions

**Session A — semantics and correctness**

- 0:00–0:35: trace one token through dense, GLU, and SwiGLU FFNs.
- 0:35–1:15: derive shapes, parameter counts, and parameter-matched widths.
- 1:15–2:15: implement activations and both reference FFNs.
- 2:15–3:00: write parity, gradient-free inference, and malformed-shape tests.

**Session B — inference behavior**

- 0:00–0:45: integrate SwiGLU into the cumulative decoder block.
- 0:45–1:30: implement the fused pointwise stage and inspect allocations.
- 1:30–2:20: benchmark prefill-like and decode-like `M` values across dtypes.
- 2:20–3:00: explain results with bytes moved, FLOPs, and kernel-launch count.

## Experiments and deliverables

Commit:

- `tests/test_ffn.py` with float64 formula tests and float32/float16 parity tests;
- `benchmarks/bench_ffn.py` reporting shape, dtype, latency, effective TFLOP/s, and peak allocated bytes;
- a parameter-count table comparing dense `4d` and SwiGLU `8d/3`;
- an ablation that times three separate pointwise operations versus the fused gate operation; and
- `notes/module09.md` explaining why a speedup at large `M` may disappear at `M=1`.

Use warmups, synchronized timings, medians, and fixed seeds. Preserve raw benchmark output.

## Common failure modes

- Swapping `gate_proj` and `up_proj` changes outputs even though shapes still match.
- Applying SiLU after the product is not SwiGLU.
- Choosing an arbitrary intermediate width makes checkpoint weights impossible to load.
- A silent broadcast, especially an extra singleton dimension, produces plausible but wrong tensors.
- Comparing fused low-precision output directly to itself hides correlated errors; keep an independent float32/64 oracle.
- Materializing both expanded branches for every token can make memory traffic dominate.
- Assuming FFN sparsity from the word “gate” is incorrect: ordinary SwiGLU still evaluates every channel.

## Mastery gate

Given `B=4, T=128, d=4096, f=11008` and a dtype, derive every intermediate shape, parameter count, approximate FLOPs, and activation-memory footprint without running code. Then load random weights into your reference and optimized implementations and demonstrate agreement under a justified tolerance for at least five shapes, including `M=1`. You pass when you can diagnose a deliberately swapped projection from outputs and state-dict names alone.

## Primary sources

- Shazeer, [“GLU Variants Improve Transformer”](https://arxiv.org/abs/2002.05202).
- Hendrycks and Gimpel, [“Gaussian Error Linear Units”](https://arxiv.org/abs/1606.08415).
- PyTorch, [`SiLU` definition and shape contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html).
- Hugging Face Transformers, [current `LlamaMLP` implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py).

## Code paths to read

Your paths are `reference/activations.py`, `reference/ffn.py`, `kernels/swiglu.py`, `reference/decoder_block.py`, `tests/test_ffn.py`, and `benchmarks/bench_ffn.py` under `inference-lab/`.

## OSS bridge

In Transformers, find `LlamaMLP` and map `gate_proj`, `up_proj`, and `down_proj` to your equations. In vLLM, inspect [`vllm/model_executor/layers/activation.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py) and model-specific MLP classes; distinguish pointwise fusion from quantized linear kernels.

## Next

Module 10 adds the token-mixing half of the decoder block: masked multi-head causal attention.
