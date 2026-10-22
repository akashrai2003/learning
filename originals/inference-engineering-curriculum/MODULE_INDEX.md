# Module Index

Each module is designed for approximately 8–12 focused hours. “Artifact” is the
minimum durable output; the module itself contains a larger implementation
ladder and mastery gate.

## Orientation and foundations

| # | Module | Minimum artifact |
|---:|---|---|
| 01 | [Inference system map](modules/01-inference-system-map.md) | Annotated request path from API to generated token |
| 02 | [Reproducible stack and toolchain](modules/02-reproducible-stack-and-toolchain.md) | Machine-readable environment manifest and rebuild procedure |
| 03 | [Tensor math, shapes, and `einsum`](modules/03-tensor-math-shapes-and-einsum.md) | Shape-checked tensor algebra exercises and tests |
| 04 | [Numerical formats and error](modules/04-numerical-formats-and-error.md) | Floating-point/rounding/error exploration notebook or script |
| 05 | [Performance modeling and roofline](modules/05-performance-modeling-and-roofline.md) | Prefill/decode FLOP-byte model with measured comparison |

## Transformer and modern-model architecture

| # | Module | Minimum artifact |
|---:|---|---|
| 06 | [Transformer forward pass](modules/06-transformer-forward-pass.md) | Framework-light decoder block with parity tests |
| 07 | [Tokenization, embeddings, and logits](modules/07-tokenization-embeddings-and-logits.md) | Token-to-logit path with tokenizer boundary tests |
| 08 | [Normalization, residuals, and stability](modules/08-normalization-residuals-and-stability.md) | RMSNorm/LayerNorm and residual-topology experiments |
| 09 | [FFN activations and SwiGLU](modules/09-ffn-activations-and-swiglu.md) | Gated FFN with parameter/FLOP analysis |
| 10 | [MHA, masking, and causal attention](modules/10-mha-masking-and-causal-attention.md) | Explicit attention oracle and mask test matrix |
| 11 | [Positional encoding, RoPE, and scaling](modules/11-positional-encoding-rope-and-scaling.md) | RoPE implementation and long-position experiments |
| 12 | [MQA, GQA, and KV cache](modules/12-mqa-gqa-and-kv-cache.md) | Cached decode equal to full-sequence decode |
| 13 | [Mixture of experts](modules/13-mixture-of-experts.md) | Toy top-k routed MoE and load-balance report |
| 14 | [Long context and memory](modules/14-long-context-and-memory.md) | Full/local/hybrid attention cost and behavior comparison |
| 15 | [Multimodal inference](modules/15-multimodal-inference.md) | Vision-language tensor-interface and profiling dossier |
| 16 | [SSM, recurrent, and hybrid models](modules/16-ssm-recurrent-and-hybrid-models.md) | Recurrent-state implementation and cache comparison |
| 17 | [Modern-model architecture forensics](modules/17-modern-model-architecture-forensics.md) | Config/checkpoint/source-derived model dossier |

## Quantization and low-precision inference

| # | Module | Minimum artifact |
|---:|---|---|
| 18 | [Quantization fundamentals](modules/18-quantization-fundamentals.md) | Scalar/vector quantizer with derived error behavior |
| 19 | [Calibration, error metrics, and evaluation](modules/19-calibration-error-metrics-and-evaluation.md) | Reproducible operator-to-task evaluation harness |
| 20 | [INT8 static, dynamic, and SmoothQuant](modules/20-int8-static-dynamic-and-smoothquant.md) | INT8 and smoothing comparison across distributions |
| 21 | [Weight-only INT8/INT4 and packing](modules/21-weight-only-int8-int4-and-packing.md) | Grouped quantizer, bit packer, and fused-reference linear |
| 22 | [GPTQ and second-order PTQ](modules/22-gptq-second-order-ptq.md) | Small Hessian-aware quantization implementation |
| 23 | [AWQ](modules/23-awq-activation-aware-quantization.md) | Activation-aware search/scaling experiment |
| 24 | [Low-bit formats, NF4, codebooks, and GGUF](modules/24-low-bit-formats-nf4-codebooks-and-gguf.md) | Format/codebook/layout comparison and file inspection |
| 25 | [FP8 training and inference](modules/25-fp8-training-and-inference.md) | FP8 emulation, scaling, and accumulation study |
| 26 | [Microscaling, MXFP4, and NVFP4](modules/26-microscaling-mxfp4-and-nvfp4.md) | Block-scale overhead/error/kernel-requirement report |
| 27 | [KV-cache quantization](modules/27-kv-cache-quantization.md) | Quantized-cache decode and long-context quality test |
| 28 | [Quantized kernel and system integration](modules/28-quantized-kernel-and-system-integration.md) | Dispatchable quantized operator with fallback and benchmark |

## Attention, decoding, and scheduling

| # | Module | Minimum artifact |
|---:|---|---|
| 29 | [Attention IO complexity and online softmax](modules/29-attention-io-complexity-and-online-softmax.md) | Tiled online-softmax attention prototype |
| 30 | [FlashAttention evolution](modules/30-flashattention-evolution.md) | Algorithm comparison and one tiled implementation |
| 31 | [PagedAttention and memory management](modules/31-pagedattention-and-memory-management.md) | Paged KV allocator/addressing simulator |
| 32 | [Decoding, sampling, and logits processing](modules/32-decoding-sampling-and-logits-processing.md) | Tested deterministic and stochastic decoding library |
| 33 | [Speculative decoding](modules/33-speculative-decoding.md) | Exact speculative decoder plus acceptance measurements |
| 34 | [Continuous batching, chunked prefill, and prefix cache](modules/34-continuous-batching-chunked-prefill-and-prefix-cache.md) | Event-driven scheduler/cache simulator |

## GPU, kernels, compilers, and builds

| # | Module | Minimum artifact |
|---:|---|---|
| 35 | [GPU architecture and Blackwell](modules/35-gpu-architecture-and-blackwell.md) | Target-GPU resource/instruction capability dossier |
| 36 | [CUDA execution, memory, and streams](modules/36-cuda-execution-memory-and-streams.md) | CUDA microbenchmark suite with synchronization checks |
| 37 | [GEMM and Tensor Cores](modules/37-gemm-and-tensor-cores.md) | Tiled GEMM progression and throughput analysis |
| 38 | [Tiling, pipelining, and fusion](modules/38-tiling-pipelining-and-fusion.md) | Fused/pipelined kernel with profiler evidence |
| 39 | [Triton kernels](modules/39-triton-kernels.md) | Softmax, normalization, and/or matmul Triton kernels |
| 40 | [CUTLASS and CuTe](modules/40-cutlass-and-cute.md) | Modified CUTLASS/CuTe example and layout explanation |
| 41 | [PyTorch custom operators](modules/41-pytorch-custom-operators.md) | Packaged C++/CUDA op with dispatcher and `opcheck` tests |
| 42 | [Dynamo, Inductor, and Triton compiler stack](modules/42-compiler-stack-dynamo-inductor-and-triton.md) | Graph/lowering/generated-code trace and custom-op integration |
| 43 | [Build, JIT, Ninja, CMake, and ABI](modules/43-build-jit-ninja-cmake-and-abi.md) | Reproduced and diagnosed build/link/JIT failure |
| 44 | [Profiling, benchmarking, and correctness](modules/44-profiling-benchmarking-and-correctness.md) | Reproducible performance investigation with regression gates |

## Distributed runtimes and open source

| # | Module | Minimum artifact |
|---:|---|---|
| 45 | [Distributed inference parallelism](modules/45-distributed-inference-parallelism.md) | Communication/memory model plus multi-rank experiment |
| 46 | [vLLM internals and contributions](modules/46-vllm-internals-and-contributions.md) | End-to-end call-path map and scoped patch proposal |
| 47 | [SGLang internals and contributions](modules/47-sglang-internals-and-contributions.md) | Runtime/RadixAttention source walk and patch proposal |
| 48 | [llama.cpp/ggml internals and contributions](modules/48-llamacpp-ggml-internals-and-contributions.md) | GGUF-to-backend graph walk and patch proposal |

## Cross-cutting documents

- [Roadmap](ROADMAP.md)
- [Concept map](CONCEPT_MAP.md)
- [Lab environment](LAB_ENVIRONMENT.md)
- [Local RTX 5060 Ti track](LOCAL_RTX_5060TI_TRACK.md)
- [Benchmarking protocol](BENCHMARKING.md)
- [Open-source contribution playbook](OPEN_SOURCE.md)
- [Extensions after the core](EXTENSIONS_AFTER_CORE.md)
- [Capstones](projects/README.md)
- [Primary sources](resources/PRIMARY_SOURCES.md)
- [Paper-note template](templates/PAPER_NOTE.md)
- [Benchmark-report template](templates/BENCHMARK_REPORT.md)
- [Source-walk template](templates/SOURCE_WALK.md)
