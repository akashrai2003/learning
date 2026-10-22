# Extensions After the 48-Module Core

The core is deliberately bounded around autoregressive LLM/VLM inference and
the systems needed to contribute to its major runtimes. The following tracks
extend the same habits—derive, implement, test, measure, inspect source—into
adjacent areas. Choose one after Module 48 rather than diluting the first six
months.

## Track A — Image and video generation systems

Learn in this order:

1. variational autoencoders and latent-space compute;
2. diffusion forward/reverse processes and noise schedules;
3. score matching, velocity prediction, and flow matching;
4. U-Net versus Diffusion Transformer architectures;
5. classifier-free guidance and batched conditional/unconditional execution;
6. text encoders, conditioning, cross-attention, and prompt weighting;
7. sampler/solver step count, quality, and stability;
8. attention and convolution kernels for spatial/video tensors;
9. quantization and mixed precision across encoder, denoiser, and decoder;
10. caching, sequence/first-block reuse, parallelism, and serving.

Build a small latent diffusion inference loop with swappable schedulers. Measure
latency by text encoding, denoising step, and VAE decode; then profile attention,
MLP, and convolution shapes. Implement one fused normalization/activation or
attention kernel, and quantify quality changes using a fixed prompt/seed suite.

Primary starting points:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Hugging Face Diffusers repository](https://github.com/huggingface/diffusers)

## Track B — Speech, audio, and streaming multimodality

Learn:

1. waveform, spectrogram, mel, and codec-token representations;
2. convolutional and Transformer encoders;
3. CTC, transducer, attention-decoder, and seq2seq ASR;
4. neural audio codecs and discrete acoustic tokens;
5. autoregressive and diffusion/flow TTS;
6. streaming chunks, lookahead, endpointing, and state caches;
7. real-time factor, first-audio latency, jitter, and quality metrics;
8. batching variable-duration media and CPU preprocessing;
9. quantization of encoders, decoders, and vocoders;
10. duplex speech-to-speech scheduling.

Capstone: a streaming ASR or TTS service whose state transitions and latency
budget are explicit. Measure first partial result/audio, real-time factor,
memory, and quality under concurrent streams.

Starting sources:

- [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [RNN Transducer](https://arxiv.org/abs/1211.3711)
- [Whisper](https://arxiv.org/abs/2212.04356)
- [EnCodec](https://arxiv.org/abs/2210.13438)

## Track C — Training/inference co-design

Inference engineers benefit from understanding how training choices create
runtime constraints:

1. reverse-mode autodiff and saved tensors;
2. optimizer state and mixed-precision training;
3. activation checkpointing;
4. data, tensor, pipeline, sequence/context, and expert parallel training;
5. ZeRO/FSDP sharding;
6. quantization-aware training and distillation;
7. LoRA and adapter serving;
8. RL/post-training rollout engines and training-serving mismatch;
9. checkpoint formats, conversion, and sharded loading;
10. architecture co-design for cache, quantization, and speculative decoding.

Extend the low-bit workbench with fake-quant/QAT and compare a PTQ model, a
distilled model, and a QAT model under the same runtime kernels. Then trace a
rollout from serving engine back into a training batch.

Starting sources:

- [GPipe](https://arxiv.org/abs/1811.06965)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
- [LoRA](https://arxiv.org/abs/2106.09685)

## Track D — Compiler depth

Move below `torch.compile` usage:

1. graph capture and Python guards;
2. operator decomposition and functionalization;
3. shape systems and symbolic dimensions;
4. high-level, loop, tile, and instruction IRs;
5. rewrite legality and numerical semantics;
6. scheduling, layout, fusion, and buffer planning;
7. MLIR and LLVM basics;
8. Triton lowering and backend code generation;
9. autotuning and cost models;
10. compiler debugging, minimization, and bisecting.

Capstone: define a tiny tensor IR and lower elementwise/reduction/matmul
expressions into a GPU schedule or a restricted CUDA/Triton subset. The goal is
not a production compiler; it is understanding where shapes, layouts, memory,
and target instructions become explicit.

Starting sources:

- [MLIR documentation](https://mlir.llvm.org/docs/)
- [LLVM documentation](https://llvm.org/docs/)
- [TorchDynamo deep dive](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html)
- [Triton repository](https://github.com/triton-lang/triton)

## Track E — Pruning, sparsity, and distillation

Quantization changes representation; pruning changes which values or structures
exist. Learn:

1. unstructured magnitude pruning and retraining;
2. one-shot LLM pruning such as SparseGPT and Wanda;
3. N:M, block, channel, head, layer, and expert sparsity;
4. hardware-supported structured sparsity contracts;
5. sparse storage formats and metadata;
6. sparse GEMM/GEMV and break-even density;
7. activation sparsity and dynamic routing;
8. structured model surgery versus masking zeros;
9. knowledge distillation and teacher/student objectives;
10. joint pruning, quantization, and kernel evaluation.

Build a small pruning workbench that reports logical sparsity, actual serialized
and resident bytes, quality, and measured speed. A dense kernel multiplying
stored zeros is not an inference optimization. Prove that the selected sparse
layout reaches a compatible kernel and find the density/shape at which it wins.

Starting sources:

- [SparseGPT](https://arxiv.org/abs/2301.00774)
- [Wanda](https://arxiv.org/abs/2306.11695)
- [Learning both Weights and Connections](https://arxiv.org/abs/1506.02626)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [cuSPARSELt documentation](https://docs.nvidia.com/cuda/cusparselt/)

## Track F — Non-NVIDIA and portable inference

Repeat selected kernel/runtime modules on one alternative backend:

- AMD ROCm/HIP;
- Apple Metal;
- CPU SIMD and AMX;
- WebGPU;
- TPU/XLA;
- an accelerator supported by a vLLM hardware plugin.

Map the common semantic operator to each memory hierarchy, execution model,
precision support, compiler, kernel library, and profiler. Portability does not
mean hiding every difference behind one lowest-common-denominator kernel.

llama.cpp/ggml is a particularly useful laboratory because backend abstraction
and quantized CPU/GPU execution are visible in one project.

## Track G — Production fleet and networking

For cluster-scale service, add:

1. queueing theory and capacity planning;
2. load balancing with cache affinity;
3. admission, priorities, quotas, and multi-tenancy;
4. prefill/decode disaggregation;
5. hierarchical/distributed KV caches;
6. topology-aware placement and collective cost;
7. autoscaling and cold starts;
8. failures, retries, idempotency, and rolling upgrades;
9. observability, tracing, and SLO/error budgets;
10. energy, cost, utilization, and workload forecasting.

Build a discrete-event fleet simulator before claiming a distributed design.
Calibrate it with single-node results and state which network/storage assumptions
are measured versus hypothetical.

## Choosing an extension

Choose based on the work you want reviewers to associate with you:

- novel kernels and hardware features → A, D, or F;
- multimodal product performance → A or B;
- model/runtime co-design → C;
- compression/model surgery → E;
- portable backend work → F;
- serving schedulers and infrastructure → G;
- broad runtime contributions → combine one domain track with one target
  repository, but only after completing its core module.

Use the same completion contract as the core: explain, derive, implement,
measure, inspect, and defend.
