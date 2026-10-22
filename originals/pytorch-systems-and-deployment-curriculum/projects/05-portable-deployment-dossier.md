# Project 05 — Portable Deployment Dossier

**Window:** Weeks 32–41 · **Prerequisites:** Modules 51–58

Take one validated model component or compact model from the earlier projects
and produce a deployment dossier instead of a collection of conversion commands.
The essential question is: *what executable contract is preserved for which
future inputs and target runtime?*

## Required artifact path

- Eager PyTorch reference and a fixed/dynamic input contract.
- `torch.export` artifact with explicitly stated range/shape constraints.
- ONNX artifact and ONNX Runtime execution; inspect nodes/providers and test
  parity across representative inputs.
- Choose **one** target branch: Torch-TensorRT/TensorRT on NVIDIA, AOTInductor
  native packaging, or ExecuTorch on an actual/emulated edge target.
- Document unsupported operators, graph partition/fallbacks, runtime libraries,
  artifact serialization format, and cold versus warm behavior.

## Evidence matrix

For each route, report artifact size, export/build time, load time, peak memory,
steady-state latency/throughput, output/task parity, precision, dynamic-shape
coverage, hardware/runtime versions, and limitations. A route that fails
cleanly with a useful diagnosis is still a valid result.

## Mastery gate

Given a new input shape or a minor model refactor, predict whether each artifact
should accept it and why. If it fails, localize the boundary: program capture,
operator translation, engine profile, execution provider, delegate partition,
or runtime ABI.

**Inference bridge:** Inference 28, 42, and 44. Pair TensorRT-LLM investigation
with vLLM/SGLang source work rather than using benchmark headlines as the sole
comparison.
