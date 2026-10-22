# Roadmap: 24 Weeks, 48 Modules, Six Capstones

## Target and workload

The deep path assumes approximately 21 focused hours per week:

- five weekday sessions of three hours;
- one six-hour weekend block;
- 24 weeks, or roughly 500 hours in total.

Use the 16-week path only if you can already implement a Transformer forward
pass, are comfortable with linear algebra and floating-point behavior, and can
read medium-sized C++ codebases. It covers the same modules at three per week,
but leaves less time for source reading and contribution work.

## The 24-week path

| Week | Modules | Implementation milestone | Reading/review milestone |
|---:|---|---|---|
| 1 | 01–02 | Capture a reproducible host/GPU/software manifest and trace one request through an engine | Explain driver, toolkit, runtime, framework, engine, and kernel-library boundaries |
| 2 | 03–04 | Build tensor-shape assertions and a floating-point/quantization sandbox | Derive ULP, relative error, accumulation, and broadcast behavior |
| 3 | 05–06 | Implement a small decoder block and create prefill/decode roofline estimates | Annotate every tensor shape and byte/FLOP estimate |
| 4 | 07–08 | Add tokenizer boundary tests, embeddings, RMSNorm, residual variants, and logits | First production source walk and first documentation-sized OSS contribution candidate |
| 5 | 09–10 | Add SwiGLU and masked MHA with an explicit attention oracle | Derive parameter, FLOP, and activation-memory formulas |
| 6 | 11–12 | Add RoPE, GQA, and a growing KV cache | Validate position handling and cache equivalence token by token |
| 7 | 13–14 | Implement a toy top-k MoE and one long-context strategy | Compare dense/MoE and full/local attention resource use |
| 8 | 15–16 | Trace one vision-language model and one recurrent/SSM hybrid | Produce architecture dossiers with exact tensor interfaces |
| 9 | 17–18 | Reconstruct a current model from config and build a scalar quantizer | Capstone 1 review; begin quantization error notebook |
| 10 | 19–20 | Build calibration/evaluation harness and INT8/SmoothQuant experiment | Separate operator error from end-task degradation |
| 11 | 21–22 | Implement grouped INT4 packing/dequantization and a small GPTQ solver | Explain when metadata or dequantization dominates |
| 12 | 23–24 | Implement AWQ-style scaling and compare NF4/GGUF-style codebooks | Capstone 2 design review |
| 13 | 25–26 | Emulate FP8 and block-scaled FP4; inspect native hardware constraints | Derive range/precision and scale-overhead trade-offs |
| 14 | 27–28 | Quantize KV cache and integrate a quantized linear dispatch path | Capstone 2 benchmark and quality report |
| 15 | 29–30 | Implement online softmax and a tiled exact-attention prototype | Explain FlashAttention by IO complexity, not by slogan |
| 16 | 31–32 | Build a paged cache simulator and complete sampling/logits tests | Audit determinism, stop conditions, penalties, and numerical stability |
| 17 | 33–34 | Add speculative decoding, continuous batching, chunked prefill, and prefix reuse to a simulator | Derive acceptance and scheduling trade-offs |
| 18 | 35–36 | Write basic CUDA kernels and inspect PTX/SASS on the target GPU | Map warps, blocks, SM resources, memory spaces, streams, and synchronization |
| 19 | 37–38 | Implement tiled GEMM and a fused epilogue; measure roofline position | Capstone 3 design and correctness review |
| 20 | 39–40 | Reimplement selected kernels in Triton and CUTLASS/CuTe | Compare abstraction, generated code, dispatch, and portability |
| 21 | 41–42 | Package a CUDA custom op and trace `torch.compile` lowering to generated kernels | Document dispatcher, FakeTensor/meta, graph breaks, guards, and caches |
| 22 | 43–44 | Reproduce a JIT/build failure; profile engine and kernels systematically | Capstone 3 benchmark report; establish regression thresholds |
| 23 | 45–46 | Model parallel communication and complete a vLLM call-path/source walk | Capstone 4: small serving engine; prepare first scoped upstream patch |
| 24 | 47–48 | Complete SGLang and llama.cpp/ggml source walks; choose one contribution track | Capstones 5–6 proposal, patch, tests, benchmarks, and reviewer-ready write-up |

The capstones overlap the modules. Do not postpone all integration until Week
24.

## Phase gates

### Gate A — foundations, end of Week 3

Without notes, you can:

- trace `[batch, sequence, hidden]` through a decoder block;
- identify which operations dominate parameters, FLOPs, and bytes;
- distinguish driver compatibility from toolkit and binary ABI compatibility;
- estimate whether a prefill or decode operation is compute- or bandwidth-bound;
- explain why lower precision can increase speed, decrease it, or change nothing.

### Gate B — architecture, end of Week 9

You can implement a cached decoder that agrees with full-sequence decoding,
describe RoPE and GQA algebraically, and reconstruct a model graph from its
configuration and checkpoint tensor names. Your architecture dossier must
include unsupported/fallback operations and memory estimates, not just a block
diagram.

### Gate C — quantization, end of Week 14

Given a model and hardware target, you can justify:

- format and granularity;
- symmetric/asymmetric and static/dynamic choices;
- calibration data and objective;
- accumulation precision;
- packing layout and kernel availability;
- quality, latency, throughput, and memory acceptance thresholds.

You must have at least one counterexample where a smaller checkpoint is not
faster.

### Gate D — attention/runtime, end of Week 17

You can derive online softmax, explain exact attention's IO cost, model KV-cache
block allocation, and implement a scheduler simulator that reports TTFT, TPOT,
goodput, fragmentation, and starvation. You can distinguish prefix caching,
chunked prefill, speculative decoding, and continuous batching by the resource
each changes.

### Gate E — GPU/compiler, end of Week 22

You can move from a profiler symptom to a falsifiable kernel hypothesis; read
basic PTX and compiler output; explain occupancy without treating it as a goal;
package and test a custom operator; and diagnose architecture flags, JIT cache,
Ninja parallelism, dynamic linking, or Python/C++ ABI failures.

### Gate F — contributor readiness, end of Week 24

You can navigate one target runtime without global text search for every step,
produce a minimal reproducer and benchmark, identify the correct subsystem and
maintainers, write tests before the optimization, and submit a patch that
separates correctness from performance claims.

## Sixteen-week accelerated mapping

Study three modules per week:

```text
W01  01–03    W05  13–15    W09  25–27    W13  37–39
W02  04–06    W06  16–18    W10  28–30    W14  40–42
W03  07–09    W07  19–21    W11  31–33    W15  43–45
W04  10–12    W08  22–24    W12  34–36    W16  46–48
```

For this route:

- preserve all mastery gates;
- reduce the number of alternative implementations, not the correctness tests;
- select four capstones rather than six;
- begin with one target hardware platform and one runtime;
- schedule the upstream contribution after the 16 weeks if review quality would
  otherwise suffer.

## Weekly operating rhythm

Each week should produce one commit containing:

1. derivations or a paper note;
2. reference implementation and tests;
3. a production source-walk note;
4. raw benchmark data plus a short interpretation;
5. one paragraph stating what evidence changed your mental model.

Use a branch per module. Tag phase gates. Never keep the only copy of an
experiment in a notebook with hidden state.

## Review system

Maintain a spaced-repetition queue of mechanisms, not trivia:

- Day 1: derive immediately after reading.
- Day 3: implement from an empty file.
- Day 7: explain while drawing shapes and memory movement.
- Day 21: solve a variant or debug an intentionally broken version.
- Day 45: connect it to a production code path or PR.

At the end of every second week, choose a random completed mastery gate and
repeat it under a 45-minute time limit.

## When to slow down

Pause the calendar and repeat a module when:

- the optimized result lacks a trusted oracle;
- measurements vary more than the claimed speedup;
- you cannot explain why the bottleneck moved;
- quality is reported from a single prompt;
- an implementation works only at one convenient shape;
- you can invoke an API but cannot locate its dispatch and fallback logic.

Finishing later with an auditable portfolio is more valuable than finishing all
48 files on schedule.

