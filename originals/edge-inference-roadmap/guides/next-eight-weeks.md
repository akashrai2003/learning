# Optional weeks 25–32: choose one specialization

Do not attempt every track at once. Retain the same daily structure: recall, focused reading, implementation, verification and a note. Each week use six build sessions and a seventh reconstruction/review session. These are extension outlines, not additional numbered daily assignments.

## Track A: ARM CPU kernels

| Week | Work | Evidence |
|---|---|---|
| 25 | Establish supported CPU features and inspect generated assembly | Feature matrix and one explained inner loop |
| 26 | Optimize packed quantized dot products | Scalar parity and shape sweep |
| 27 | Explore weight packing and data layout | Repacking cost versus repeated-use benefit |
| 28 | Implement cache blocking for a selected matmul workload | Working-set explanation and measured comparison |
| 29 | Investigate threading and scheduling | Small/large workload crossover |
| 30 | Try a justified fusion | End-to-end benefit and numeric contract |
| 31 | Integrate one improvement into a runtime branch | Focused regression tests |
| 32 | Prepare a contribution and report | Reproducible patch or issue draft |

## Track B: NPU export and compiler deployment

| Week | Work | Evidence |
|---|---|---|
| 25 | Capture and inspect transformer graphs | Operator and shape inventory |
| 26 | Trace decomposition and lowering | Before/after graph explanation |
| 27 | Study backend quantization requirements | Calibration and quality comparison |
| 28 | Measure partition boundaries and fallback | Verified delegated operations |
| 29 | Explore supported sequence/cache configurations | Compatibility and memory table |
| 30 | Optimize preparation and repeated use | Initialization versus steady-state results |
| 31 | Minimize a real unsupported case | Small reproducible model |
| 32 | Contribute a test, fix or documentation improvement | Specific evidence-backed contribution |

## Track C: Runtime infrastructure — recommended first for your background

| Week | Work | Evidence |
|---|---|---|
| 25 | Model/session/workspace accounting | Allocation inventory and limits |
| 26 | Admission control and bounded queues | Queue-time and overload tests |
| 27 | Exact prefix reuse and invalidation | Correctness and repeated-prefix benefit |
| 28 | Simulate block-based KV allocation | Fragmentation and capacity analysis |
| 29 | Study continuous batching on your server environment | Scheduler trace with realistic requests |
| 30 | Explore speculative decoding with a supported runtime | Acceptance rate and total latency, including drafter cost |
| 31 | Study model switching and state restoration | Failure-safe transitions and memory peaks |
| 32 | Implement one small runtime improvement | Tests and an end-to-end comparison |

The simulation in week 28 is educational; implementing a general paged attention kernel is a separate project. Server batching experiments belong on an appropriate server/runtime, not necessarily the phone. CUDA is useful for this wider inference track but does not execute on the phone's Adreno or Hexagon hardware.

## Optional multimodal project

Choose speech recognition plus generation, a small image encoder plus grounded answers, or local OCR-assisted extraction. Establish standalone stage correctness before composing models. Measure preprocessing, encoder, decoder, buffering, memory transitions and task quality. Reuse the capstone's cancellation and asset policies rather than building a second application framework.
