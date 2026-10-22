# Primary Source Index

This index is a spine, not a queue to read front-to-back. Use the exact subset
assigned by each module, implement the mechanism, then return to the paper after
profiling or debugging it.

For every paper note, distinguish:

- what the paper proves or measures;
- what it assumes;
- what the released code actually implements;
- what later production runtimes changed;
- what remains unsupported on your hardware.

## Transformer and model architecture

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original
  Transformer.
- [RMSNorm](https://arxiv.org/abs/1910.07467) — normalization without
  re-centering.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) —
  gated FFN variants including SwiGLU.
- [RoFormer](https://arxiv.org/abs/2104.09864) — rotary position embedding.
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
  — multi-query attention.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
  — grouped-query attention.
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — sparse expert
  routing at scale.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — open sparse MoE
  architecture and evaluation.
- [Longformer](https://arxiv.org/abs/2004.05150) and
  [BigBird](https://arxiv.org/abs/2007.14062) — structured sparse attention.
- [Mamba](https://arxiv.org/abs/2312.00752) and
  [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060) —
  selective state spaces and structured state-space duality.
- [Jamba](https://arxiv.org/abs/2403.19887) — hybrid
  Transformer/Mamba/MoE design.

## Multimodal architecture

- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Flamingo](https://arxiv.org/abs/2204.14198)
- [LLaVA](https://arxiv.org/abs/2304.08485)
- [SigLIP](https://arxiv.org/abs/2303.15343)

Use the selected model's official processor/tokenizer configuration and source
alongside its paper. Image preprocessing, placeholder expansion, and batching
are part of inference semantics.

## Modern architecture forensics

These reports are useful comparison targets rather than a chronological reading
requirement:

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770)
- [Gemma 4 official model page](https://deepmind.google/models/gemma/gemma-4/)

Always pair a report with model config, checkpoint tensor index, tokenizer or
processor files, and canonical implementation.

## Quantization

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
  — affine quantization foundations.
- [LLM.int8()](https://arxiv.org/abs/2208.07339) — outlier-aware 8-bit matrix
  multiplication.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) — migration of activation
  difficulty into weights.
- [GPTQ](https://arxiv.org/abs/2210.17323) — second-order post-training
  weight quantization.
- [AWQ](https://arxiv.org/abs/2306.00978) — activation-aware weight
  quantization.
- [SpQR](https://arxiv.org/abs/2306.03078) — sparse outlier handling at very
  low precision.
- [QLoRA](https://arxiv.org/abs/2305.14314) — NF4 and double quantization;
  training-centered, but important for codebook reasoning.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA Transformer Engine NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [KIVI](https://arxiv.org/abs/2402.02750) and
  [KVQuant](https://arxiv.org/abs/2307.13304) — KV-cache quantization.
- [GGUF specification and canonical source](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [compressed-tensors](https://github.com/vllm-project/compressed-tensors) —
  production quantization configuration and compression library.

## Attention, cache, and decoding

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [PagedAttention/vLLM paper](https://arxiv.org/abs/2309.06180)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [EAGLE](https://arxiv.org/abs/2401.15077)

When algorithms claim exact sampling, reproduce the acceptance proof and test
the output distribution; acceptance rate alone is not correctness.

## GPU architecture and CUDA

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)
- [Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [Nsight Systems documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)

Read the guide matching the installed toolkit when exact flags, instructions,
or resource limits matter.

## Kernel languages and libraries

- [Triton language and compiler](https://triton-lang.org/main/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton repository](https://github.com/triton-lang/triton)
- [CUTLASS documentation](https://docs.nvidia.com/cutlass/)
- [CUTLASS repository](https://github.com/NVIDIA/cutlass)
- [CuTe DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
- [FlashInfer documentation](https://docs.flashinfer.ai/)
- [FlashInfer repository](https://github.com/flashinfer-ai/flashinfer)

## PyTorch compilation and custom operators

- [PyTorch compiler programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)
- [Dynamo deep dive](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [Custom C++ and CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [Custom operators and `torch.compile`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.custom_ops.html)
- [PyTorch repository](https://github.com/pytorch/pytorch)

## Build and binary tooling

- [Ninja manual](https://ninja-build.org/manual.html)
- [CMake documentation](https://cmake.org/documentation/)
- [GNU dynamic linker documentation](https://man7.org/linux/man-pages/man8/ld.so.8.html)
- [Python/C API stable ABI](https://docs.python.org/3/c-api/stable.html)
- [CUDA compilation with NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)

## Serving and distributed systems

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) —
  iteration-level scheduling.
- [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked-prefill-aware
  scheduling.
- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
  — prefill/decode disaggregation and goodput.
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html)

## Runtime and contribution references

- [vLLM documentation](https://docs.vllm.ai/)
- [vLLM contribution guide](https://docs.vllm.ai/en/latest/contributing/)
- [SGLang documentation](https://docs.sglang.io/)
- [SGLang contribution guide](https://docs.sglang.io/docs/developer_guide/contribution_guide)
- [llama.cpp repository](https://github.com/ggml-org/llama.cpp)
- [llama.cpp contribution guide](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
- [llama.cpp model-port guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)

Consult the live policy on the day you contribute. CI, review, signoff, and
AI-use rules can change.

## Books and course-length supplements

These are secondary learning aids, not substitutes for primary sources:

- *Computer Architecture: A Quantitative Approach*, Hennessy and Patterson.
- *Programming Massively Parallel Processors*, Hwu, Kirk, and Hajj.
- *Numerical Linear Algebra*, Trefethen and Bau.
- [Stanford CS149: Parallel Computing](https://gfxcourses.stanford.edu/cs149/)
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [GPU MODE lectures and tracks](https://github.com/gpu-mode/lectures)
