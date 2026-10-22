# The combined roadmap

The sequence is **semantics → correct model → numerical tradeoffs → compiler/kernel behavior → serving/distribution → deployment → evidence**. C++ grows alongside PyTorch, and career work uses the same projects throughout.

## Phase map

| Approximate period | Weeks | What you learn | Evidence at the end |
|---|---|---|---|
| Month 1 | 1–4 | Reproducibility, tensor layouts, numerics, C++ basics/ownership, autograd | Measured baseline report and foundation checks |
| Month 2 | 5–8 | State/checkpoints, CMake, tiny training loop, bindings, Transformer and KV cache | Reloadable model and verified cached decoder |
| Month 3 | 9–12 | Checkpoint parity, native operators, quality evaluation, INT8; buffer 1 | Model/evaluation dossier and first quantization report |
| Month 4 | 13–16 | INT4, TorchAO, low-bit survey, FX/compile, guards and graph breaks | Quantization choice and reproducible compiler experiment |
| Month 5 | 17–20 | Triton, custom ops, dispatcher, CUDA, tiled GEMM/fusion | One integrated operator with a defensible benchmark |
| Month 6 | 21–24 | Online softmax, efficient attention, graph capture, cache/scheduling; buffer 2 | Profiled path and deterministic serving simulator |
| Month 7 | 25–28 | Runtime internals, collectives, distributed semantics/recovery, export/ONNX | Runtime and portable-artifact dossiers |
| Month 8 | 29–32 | ExecuTorch, native Android/JNI, sustained edge behavior, architecture breadth | Edge/host baseline and a focused model adaptation |
| Month 9 | 33–36 | Runtime choices, capstone integration, contribution; buffer 3 | Integrated flagship and reviewable patch |
| Final three weeks | 37–39 | Regression, reproducible release, portfolio, assessment | Case studies and a targeted next specialization |

“Month” is approximate: 39 weeks is about nine calendar months, not ten separate month-long blocks. The final three weeks complete that total. [The weekly index](WEEK_INDEX.md) gives the authoritative order and all daily assignments.

## Why this order

Tensor shapes, precision, state and ownership precede optimization. The decoder becomes the shared oracle for cache, quantization and kernel work. Compiler behavior comes before native integration so an operator's shape and mutation contract is understood. Production-runtime source reading follows a small cache/scheduler implementation. Device-specific deployment arrives after there is a correct, exportable model and a useful measurement harness.

A tiny native binding is introduced in week 6, then registered/custom operators are formalized in week 18. Week 17 introduces enough GPU execution vocabulary for Triton; week 19 develops CUDA's native programming model. This avoids requiring deep CUDA before a first useful Triton experiment.

## Career work runs alongside the technical sequence

Weeks 1–4 produce a baseline report while you clarify role fit, improve the resume and begin selected applications. Weeks 5–8 add coding/design practice, source reading and better evidence. Weeks 9–16 emphasize evaluation, reliability, interviews and pipeline diagnosis. Thereafter, a smaller maintenance lane preserves interviews and contribution follow-through while technical depth increases.

This plan gives career work **four hours/week initially**, less than the original career-first schedule. It preserves early action but does not preserve that schedule's activity volume or imply the same probability of an offer within 3–4 months. If changing jobs quickly becomes the priority, use the career-priority allocation in [STUDY_SYSTEM.md](STUDY_SYSTEM.md) and extend technical deadlines.

## Time accounting

| Component | Weeks 1–16 | Weeks 17–39 | Total |
|---|---:|---:|---:|
| Technical work | 192 h | 322 h | 514 h |
| Career work | 64 h | 46 h | 110 h |
| Weekly review | 16 h | 23 h | 39 h |
| All work | 272 h | 391 h | **663 h** |

The three buffer weeks are included, not extra. Buffers repair earlier gates; they do not introduce a new project if everything is already passing. A clean buffer week can be lighter than its reserved hours.

At 15 hours/week, the same workload is about 45 weeks; at 20 hours/week, about 34 weeks before allowing for prerequisite sequencing and real-life interruptions. At the recommended 17-hour rhythm it is 39 weeks. These are planning estimates, not guarantees.

## Six months versus nine months

Six months is about 26 weeks. The complete 663-hour scope would require about **25.5 hours/week**, above your chosen budget. For six months at 17 hours/week, choose a narrower serving/PyTorch core:

**Keep original combined weeks 1–11, 13–18, 22–23, 25–26, 28–29, 34, 37 and 39** (26 weeks total). Renumber them for your calendar. In week 22 use the verified Triton/custom-op route; in week 34 choose the serving capstone. Retain all relevant correctness gates. If a skipped buffer is needed, extend the calendar.

That route postpones hands-on native CUDA/GEMM and online-attention depth, distributed training depth, Android lifecycle work, architecture surveys and the dedicated contribution sprint. It is **not** equivalent to completing the recommended combined scope. Your stated broad goal fits the 39-week route better.

## Dependency and scope rules

Do not optimize a decoder with unexplained parity failures. Do not benchmark a quantized format with undefined packing or calibration. Do not move an operator to another runtime before its supported shapes and state are explicit. Do not report a device/backend result from a simulator or a source walk.

If a gate fails, repair it in the next session and shift dependent tasks. Keep independent career tasks moving. The [coverage map](COVERAGE.md) shows where a module is only surveyed and which work is deliberately left for [extensions](EXTENSIONS.md).
