# Module 25 — Torchvision and Vision Model Engineering

**Phase:** 4 — PyTorch model engineering and optimization  
**Time:** 9 focused hours  
**Prerequisites:** tensors, convolutions/attention, data loading, and Modules 22–24.

## Existing inference curriculum bridge

Pair this with [03 — Tensor math, shapes, and einsum](../../inference-engineering-curriculum/modules/03-tensor-math-shapes-and-einsum.md), [15 — Multimodal inference](../../inference-engineering-curriculum/modules/15-multimodal-inference.md), [17 — Architecture forensics](../../inference-engineering-curriculum/modules/17-modern-model-architecture-forensics.md), [38 — Tiling, pipelining, and fusion](../../inference-engineering-curriculum/modules/38-tiling-pipelining-and-fusion.md), and [44 — Profiling and correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md).

## Why this matters and outcomes

Multimodal inference failures are frequently preprocessing failures disguised as model failures. A high-quality vision engineering foundation means treating image decode, color order, resize/crop, scale/range, normalization, layout, batching, and postprocessing as part of the model contract. `torchvision` provides reference models, weight-associated transforms, datasets, and operators—but each needs version and semantics audit.

You will build reproducible image pipelines, understand model input/output shapes, use weight metadata rather than guessing transforms, profile NCHW versus channels-last decisions, and create tests that protect an exported or serving system from silent image semantic drift.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P25 — Torchvision and vision model engineering**.

1. [Torchvision docs](https://docs.pytorch.org/vision/stable/) — learn the API or system contract.
2. [Torchvision repository](https://github.com/pytorch/vision) — study the underlying design, paper, or specification.
3. [ResNet paper](https://arxiv.org/abs/1512.03385) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [TorchAO source](https://github.com/pytorch/ao) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [Torchtune source](https://github.com/pytorch/torchtune) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

An RGB batch typically has shape `(B,3,H,W)`. A canonical normalized image is

\[
x'_{bchw}=\frac{x_{bchw}/255-\mu_c}{\sigma_c},
\]

but that expression hides important choices: whether decoded values begin as `uint8` `[0,255]` or float `[0,1]`, RGB versus BGR, gamma/color profile behavior, aspect-ratio policy, and antialiasing. A model’s trained preprocessing is part of its statistical input distribution. A 224×224 crop is not interchangeable with an arbitrary resize merely because the tensor shape matches.

Convolution maps `(B,C_in,H,W)` to `(B,C_out,H',W')`; a ViT patch projection maps `(B,3,H,W)` to `(B,N,D)` where \(N=HW/P^2\) for patch size \(P\) under compatible dimensions. The latter makes resolution changes token-count changes, which affects attention \(O(N^2)\), memory, positional embeddings, and later LLM visual-token budgets. Detection outputs additionally combine boxes `(N,4)`, class logits `(N,K)`, and scores; coordinate convention, image scale, and NMS threshold are therefore semantic outputs, not cosmetic postprocessing.

`torchvision.models` weight enums and their associated transforms are a strong baseline. Read their implementation in the installed version and serialize the chosen weights/transform recipe. Transforms v2 and TVTensors aim to preserve structured objects (image, boxes, masks) through compatible geometric transforms; do not apply an image-only crop independently to labels. For GPU performance, compare contiguous NCHW with channels-last experimentally; compatibility is operator/model/dtype specific, and output parity precedes speed.

## Implementation ladder in `torch-lab/`

1. **Reference:** implement `torch-lab/vision/reference_pipeline.py` using explicit decode → RGB conversion → resize/crop → float/range → normalize. Unit-test a synthetic color chart and one image with known boxes/mask.
2. **Optimized:** use a pinned torchvision classification or detection model and its published weight transforms in `torch-lab/vision/torchvision_baseline.py`. Add DataLoader workers, pinned memory, nonblocking copies, and an optional channels-last trial only with parity tests.
3. **Integrated:** package the preprocessing with model metadata for `torch.export`/ONNX or a small multimodal prefix pipeline. Benchmark decode, CPU preprocessing, H2D, forward, and postprocess separately; a fast model does not rescue a CPU-bound pipeline.

## Three focused sessions

1. **Semantic pipeline:** recreate reference transforms by hand, compare every intermediate tensor, and deliberately break RGB/BGR, range, and crop order to see prediction changes.
2. **Structured vision:** use v2 transforms or an equivalent explicit paired transform for image/box/mask data. Validate coordinate transforms visually and numerically.
3. **Systems profile:** profile batch 1 and batch N, channels-last on/off, CPU workers/pinning, and dynamic resolution. Explain bottlenecks from traces rather than GPU utilization alone.

## Experiments and deliverables

Create `preprocess_contract.yaml`, a golden input/output fixture, `test_vision_semantics.py`, `layout_benchmark.csv`, and `model_card_addendum.md`. Record decoder/library versions, source color convention, transform code/revision, input range/dtype, interpolation/antialias choice, output coordinate convention, model weights hash, and accuracy/latency impact.

## Common failure modes

- Normalizing `uint8` values as though they were `[0,1]` floats.
- Separating an image transform from its boxes/masks.
- Assuming a tensor’s shape tells you its color space or pixel range.
- Measuring a cached, already-decoded image path as end-to-end camera latency.
- Changing layout/memory format without checking custom operators and export backends.

## Mastery gate

Given an image classification or detection mismatch between Python and deployment, localize it to decode, transform, layout, model, or postprocessing with saved intermediate tensors. Then make a reproducible benchmark that identifies the dominant stage for batch 1 and batch 16.

## Primary sources, code paths, and OSS bridge

- [Torchvision documentation](https://docs.pytorch.org/vision/stable/)
- [Torchvision model weights and transforms](https://docs.pytorch.org/vision/stable/models.html)
- [Transforms v2 guide](https://docs.pytorch.org/vision/stable/transforms.html)
- [Torchvision repository](https://github.com/pytorch/vision)
- [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

Read `torchvision/models/`, `torchvision/transforms/v2/`, `torchvision/tv_tensors/`, `torchvision/ops/`, and their tests. Strong contributions include a transform parity test, a documented weight-preprocessing discrepancy, a minimal export failure, or a correctness fix in a CPU/CUDA operator; make an accuracy contract before a performance claim.

## Next

Module 26 applies the same semantic-contract discipline to streaming audio and its evolving PyTorch ecosystem.
