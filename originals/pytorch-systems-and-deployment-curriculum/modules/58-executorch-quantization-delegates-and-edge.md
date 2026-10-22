# Module 58 — ExecuTorch Quantization, Delegates, and Edge Deployment

**Phase:** 8 — Export and deployment  
**Time:** 18 focused hours  
**Prerequisites:** Modules 18–28, 24, and 57; an evaluation dataset and a named edge target or explicit emulation plan.

## Existing inference curriculum bridge

This is the edge implementation route for [18 — Quantization fundamentals](../../inference-engineering-curriculum/modules/18-quantization-fundamentals.md) through [28 — Quantized kernel and system integration](../../inference-engineering-curriculum/modules/28-quantized-kernel-and-system-integration.md), plus [15 — Multimodal inference](../../inference-engineering-curriculum/modules/15-multimodal-inference.md). The prior modules teach formats, calibration, packing, and NVFP4; this one asks whether a specific target delegate/kernel actually supports and benefits from a chosen edge quantization scheme.

## Why this matters and outcomes

At the edge, “INT8” or “4-bit” says almost nothing without the backend. Quantization changes graph operators, layouts, calibration requirements, model size, runtime memory, numerical quality, and delegate eligibility. ExecuTorch deliberately connects PyTorch/TorchAO quantization workflows to backend-specific quantizers and lowering, making a useful end-to-end testbed for the discipline that also applies to server quantization.

You will be able to choose a target-first quantization plan, run PT2E/TorchAO prepare-calibrate-convert flows where appropriate, audit delegate coverage, measure model/binary/memory/latency/quality, and make a concrete ship/defer decision instead of reporting bit width alone.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P58 — ExecuTorch quantization, delegates, and edge**.

1. [ExecuTorch quantization docs](https://docs.pytorch.org/executorch/stable/quantization-overview.html) — learn the API or system contract.
2. [ExecuTorch backend delegation](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html) — study the underlying design, paper, or specification.
3. [PyTorch quantization docs](https://docs.pytorch.org/docs/stable/quantization.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [torch.export docs](https://docs.pytorch.org/docs/stable/export.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [ExecuTorch source](https://github.com/pytorch/executorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

ExecuTorch documentation describes quantization as backend-specific. A target quantizer expresses supported precisions, granularity, static/dynamic/weight-only behavior, and operator coverage; the typical PT2E path is export, prepare with that quantizer, calibrate representative data, convert, evaluate, then lower to the target backend. The resulting Q/DQ or quantized-decomposed graph is an intermediate evidence artifact, not necessarily the runtime's final kernel representation. A delegate may fuse it into a target-specific op—or decline it and leave fallback work.

Choose calibration data from the actual input distribution and preserve preprocessing/tokenization as part of the experiment. Output error is only a diagnostic; task quality, tail cases, safety behavior where relevant, model size, peak memory, binary size, energy/latency on the target, and delegate coverage are independent metrics. XNNPACK, Core ML, QNN, Vulkan, and other backends have different hardware/support matrices. The RTX 5060 Ti can run host-side export and some desktop validation, but it is not evidence for phone NPU, mobile CPU, or embedded power behavior. Start with XNNPACK/host where it is the honest target proxy; defer device claims until a physical target exists.

Low-bit LLM work deserves extra caution. TorchAO source-level quantization and PT2E backend flows can differ, and supported qconfigs change. ExecuTorch's own LLM/export documentation distinguishes XNNPACK-style CPU paths from PT2E paths for QNN/CoreML/Vulkan. Version-pin all APIs, do not assume NVFP4 server lessons transfer to mobile delegates, and keep an FP32/BF16 reference pipeline for quality diagnosis.

## Implementation ladder in `torch-lab/`

1. **Reference:** create `torch-lab/edge/target_contract.md` and `eval_reference.py`: target device/backend, model/task/preprocessing, latency/memory/binary/quality budget, representative calibration split, and acceptance thresholds.
2. **Optimized:** create `quantize_lower.py` for a small vision/audio/encoder model using the target backend's documented quantizer. Save pre-prepare, prepared, converted, and lowered graph snapshots plus calibration/quality results.
3. **Integrated:** build `edge_deployment_dossier.md` that compares FP reference, quantized eager, exported `.pte`, and host/target runtime. Include delegate coverage, fallback cost, model and runtime binary size, arena memory, latency distribution, accuracy, and a supported/unsupported decision.

## Three focused sessions

1. **Target-first design:** read the exact backend support/quantization page, select one supported qconfig, and reject a tempting unsupported format before coding. Define calibration/evaluation splits that prevent leakage.
2. **Graph/audit loop:** prepare, calibrate, convert, lower, inspect Q/DQ/delegate transformations, and test representative plus adversarial/boundary inputs against the high-precision reference.
3. **Deployment evidence:** measure host runtime as a separate proxy, then write a device test plan covering install/build, input acquisition, warm/cold policy, power/thermal state, and failure logging. Only call the result edge performance after that target run.

## Experiments and deliverables

Commit `target_contract.md`, `quantize_lower.py`, `eval_reference.py`, `test_quantized_parity.py`, graph dumps, calibration/evaluation manifests, and `edge_deployment_dossier.md`. Record model/checkpoint/preprocessing hashes, PyTorch/TorchAO/ExecuTorch/backend revisions, qconfig/granularity/group size, calibration data summary, delegate coverage/fallbacks, graph/operator map, artifact/runtime sizes, memory arenas, cold/warm latency percentiles, task metrics/error slices, target hardware/power state, and deferred evidence.

## Common failure modes

- Selecting a mathematically appealing bit width unsupported by the backend's kernels.
- Calibrating on arbitrary random inputs or evaluating on calibration samples.
- Measuring only `.pte` weight size while ignoring runtime/delegate binary and arena memory.
- Assuming host XNNPACK timing transfers to a phone NPU or GPU.
- Calling an undelegated/fallback graph a quantized accelerator win.

## Mastery gate

Deliver an edge deployment dossier with a target-specific qconfig, reproducible calibration, quality evidence, graph/delegate audit, and complete size/memory/latency accounting. Reject or defer a configuration when coverage, target support, or quality evidence is insufficient, and explain why that is the technically correct outcome.

## Primary sources, code paths, and OSS bridge

- [ExecuTorch quantization overview](https://docs.pytorch.org/executorch/stable/quantization-overview.html)
- [ExecuTorch XNNPACK quantized lowering](https://docs.pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html)
- [ExecuTorch LLM export](https://docs.pytorch.org/executorch/stable/llm/export-llm.html)
- [ExecuTorch selective build](https://docs.pytorch.org/executorch/stable/kernel-library-selective-build.html)
- [TorchAO documentation](https://docs.pytorch.org/ao/stable/index.html)

Read the selected backend's quantizer, partitioner, operator registry, and tests in the pinned ExecuTorch tree. Productive contributions include an operator-support correction, calibration/edge-case regression, or a narrow backend coverage test; include target hardware/software evidence and avoid promising unsupported formats.

## Next

Module 59 begins the PyTorch source-tree path needed to turn these observations into upstream-quality work.
