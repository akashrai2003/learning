# Module 24 — TorchAO: Quantization, Sparsity, and Low-Bit Tensors

**Phase:** 4 — PyTorch model engineering and optimization  
**Time:** 12 focused hours  
**Prerequisites:** Modules 18–28 of the inference curriculum, PyTorch modules, and Module 22.

## Existing inference curriculum bridge

This is the PyTorch implementation companion to [18 — Quantization fundamentals](../../inference-engineering-curriculum/modules/18-quantization-fundamentals.md), [19 — Calibration and error metrics](../../inference-engineering-curriculum/modules/19-calibration-error-metrics-and-evaluation.md), [20 — INT8](../../inference-engineering-curriculum/modules/20-int8-static-dynamic-and-smoothquant.md), [24 — NF4/codebooks/GGUF](../../inference-engineering-curriculum/modules/24-low-bit-formats-nf4-codebooks-and-gguf.md), [25 — FP8](../../inference-engineering-curriculum/modules/25-fp8-training-and-inference.md), [26 — NVFP4](../../inference-engineering-curriculum/modules/26-microscaling-mxfp4-and-nvfp4.md), and [28 — Quantized integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md).

## Why this matters and outcomes

TorchAO is where modern PyTorch quantization work is concentrating. It is deliberately evolving: configuration names, supported layouts, tensor subclasses, and kernel preferences can change across releases. Learn its architecture and pin an exact TorchAO/PyTorch/CUDA tuple; do not build a career plan around a copied import line.

You will be able to derive a quantization scheme, inspect its stored representation, verify dequantization/error, establish whether its kernel is actually selected, and decide honestly whether it is a training, local inference, or export/deployment solution.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P24 — TorchAO quantization, sparsity, and low bit**.

1. [TorchAO documentation](https://docs.pytorch.org/ao/stable/) — learn the API or system contract.
2. [TorchAO repository](https://github.com/pytorch/ao) — study the underlying design, paper, or specification.
3. [GPTQ paper](https://arxiv.org/abs/2210.17323) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [TorchAO source](https://github.com/pytorch/ao) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Torchtune source](https://github.com/pytorch/torchtune) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

Affine quantization maps a real value to an integer code,

\[
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),\quad \hat x=s(q-z).
\]

Scale granularity changes the error/cost frontier: per-tensor is cheap but vulnerable to outliers; per-channel/per-group adds metadata and better local fit; dynamic activation quantization computes parameters at runtime. Weight-only quantization avoids activation conversion but needs a matmul kernel that consumes packed weights. A small serialized model is not necessarily a fast model.

TorchAO separates a flow/configuration from a low-bit tensor representation and from a primitive kernel. A `quantize_` call can replace `nn.Linear` weights with a tensor subclass carrying packed data/scales/layout; later dispatch selects a compatible operation. Therefore inspect: (1) which modules changed, (2) exact payload dtypes/shapes/bytes, (3) dequantized error, (4) dispatch/kernel path, and (5) compiled/eager compatibility. A derived dtype is a contract, not merely an integer tensor exposed by `.view()`.

For structured sparsity, density is only one part of the contract. NVIDIA-style 2:4 means two nonzeros in each fixed four-element group, requiring compressed data **and** metadata plus a supported kernel. Unstructured zeros normally do not speed dense GEMM. Combining low-bit and sparsity compounds layout/metadata restrictions; prove correctness before comparing throughput.

The legacy `torch.ao.quantization` flows are being centralized into TorchAO. Learn them enough to read existing code, but use the current TorchAO documentation and source for new work. Record the exact release because both transition status and APIs are version-sensitive.

## Implementation ladder in `torch-lab/`

1. **Reference:** write `torch-lab/quant/reference_affine.py` implementing symmetric/asymmetric per-tensor, per-row, and groupwise INT8/INT4 quantize/dequantize. Test shape tails, zero ranges, NaN policy, packing byte count, MSE, cosine similarity, and max error.
2. **Optimized:** in `torch-lab/quant/torchao_lab.py`, apply one pinned TorchAO weight-only configuration to a small decoder or vision MLP. Serialize an audit containing module names, subclass type, scales, packed storage bytes, and numerical outputs. Compare to a pure PyTorch dequantize-then-matmul oracle.
3. **Integrated:** benchmark eager and `torch.compile` paths, then one supported serving/export route. Add a 2:4 experiment only after validating the exact pattern and target kernel. Treat no speedup as informative evidence, not a reason to claim it worked.

## Three focused sessions

1. **Representation:** hand-implement affine/groupwise quantization and inspect codes/scales with a tiny matrix containing outliers and tail groups.
2. **TorchAO source walk:** pin releases, read config → tensor subclass → linear dispatch → kernel. Make one version-difference note from release tags rather than guessing compatibility.
3. **Systems experiment:** compare FP/BF16, low-bit, and low-bit-plus-sparsity across latency, throughput, model bytes, peak memory, and quality; profile actual kernel names.

## Experiments and deliverables

Commit `quant_contract.md`, `storage_audit.json`, `error_dashboard.csv`, `kernel_evidence.md`, and property tests. Each result records model revision, calibration data hash, granularity/group size, scale/zero-point policy, packing layout, target GPU/SM, TorchAO/PyTorch versions, reference tolerance, and cold/warm timings.

## Common failure modes

- Reporting theoretical 4× savings while including neither scales nor packing/alignment overhead.
- Measuring `model.state_dict()` size but not resident VRAM or runtime weight expansion.
- Feeding a TorchAO internal layout to a vLLM/llama.cpp kernel that expects another format.
- Comparing outputs after different tokenizer/prompt/template paths.
- Assuming arbitrary zeros trigger sparse tensor cores.

## Mastery gate

For a chosen config, draw the path from original `Linear.weight` through packed representation and dispatch to the invoked kernel. Reproduce its output against a reference over adversarial shapes, show an error/quality budget, and explain whether any observed speedup is compute-, bandwidth-, launch-, or conversion-bound.

## Primary sources, code paths, and OSS bridge

- [TorchAO documentation](https://docs.pytorch.org/ao/stable/)
- [TorchAO quantization architecture overview](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html)
- [TorchAO sparsity overview](https://docs.pytorch.org/ao/stable/contributing/sparsity.html)
- [TorchAO repository](https://github.com/pytorch/ao)
- [PyTorch quantization migration notice](https://docs.pytorch.org/docs/stable/quantization.html)

Read `torchao/quantization/`, `torchao/dtypes/`, `torchao/sparsity/`, `torchao/kernel/`, and their tests. Useful contributions begin with an adversarial layout test, a dispatch/fake-tensor fix, a source-linked benchmark, or documentation for an unsupported combination—never with an unverified “faster INT4” claim.

## Next

Modules 25–27 widen the same PyTorch discipline to vision, audio, and multimodal inputs.
