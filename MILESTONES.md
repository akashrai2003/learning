# Milestones and evidence gates

A week is complete when its specific gate passes, not when the date changes. These checkpoints summarize the larger dependencies. Use the weekly file for the exact daily completion checks.

| After week | Gate | Required evidence | Repair if it fails |
|---|---|---|---|
| 4 | Foundations and first public-quality report | Predict layouts; explain ownership and autograd modes; raw-to-report reproduction | Reduce to one layout, one lifetime bug and one request fixture |
| 6 | Model state and native boundary | Resume tiny training; reload state; explain binding lifetime and clean build | Fix state/buffer ownership before native model work |
| 9 | Reference model correctness | Causal attention, tokenizer/checkpoint mapping and full/cached parity; one verified native op | Compare the first divergent intermediate on a tiny input |
| 11–12 | Numerical baseline | Scalar INT8 edge cases, calibration split, error/quality definitions | Repair constants, axes, data leakage and tolerance choices in buffer 1 |
| 14 | Practical quantization | Bit-exact packing and tails; real byte count; one supported framework path | Preserve float baseline and narrow unsupported dispatch |
| 16 | Compiler behavior | Eager/compiled parity, compile-time separation, guard matrix, minimal graph-break case | Minimize to one operation and one shape change |
| 18 | Operator contract | Schema, device/layout/dtype restrictions, fake/meta behavior, clean native build | Fix registration and contract tests before GPU integration |
| 22–24 | Optimization and serving semantics | Verified kernel; profile-backed full-path report; cache/scheduler invariants | Use buffer 2 for numerics, synchronization and request accounting |
| 27 | Distributed reasoning | Two-process semantics or labeled model, topology limits, checkpoint recovery | Reduce to two ranks and one collective; no unsupported scaling claim |
| 28–29 | Artifact correctness | Declared export inputs, parity in an available second runtime, backend identity | Export a smaller supported module before a full decoder |
| 31 | Edge behavior or explicit substitute | Native/host execution, cancellation/restart/resource checks, actual target record | Keep CPU/host route; leave physical-device gate unverified |
| 34 | Integrated flagship | Reproducible main path, quality evidence, failure/recovery, decision record | Freeze new features and repair the broken contract |
| 35–36 | Contribution readiness | Reproducer, narrow patch/test, recorded checks and reviewer explanation | Use buffer 3 for scope and clean reproduction |
| 37 | Release readiness | Clean-environment minimum demo and report regeneration | Remove hidden dependencies and narrow unsupported configurations |
| 39 | Independent systems judgment | Closed-book reconstruction, three case studies, defended deployment decision | Pick the weakest gate as the next specialization's first task |

## Score each gate

- **0 — Not demonstrated:** no artifact or unable to explain the behavior.
- **1 — Assisted/partial:** works only with close guidance, one convenient case or unexplained failures.
- **2 — Independent:** explain, implement/inspect and verify the stated supported scope with evidence.
- **3 — Transfer:** solve a new variant or diagnose an unfamiliar failure using the same mechanism.

Aim for 2 on required gates. A survey topic does not require a complete implementation; it requires an accurate boundary/decision note. Hardware-dependent work can be marked `deferred — unavailable hardware`, with the host substitute scored separately. That is not the same as verifying the hardware gate.

## Performance and quality acceptance

Choose numerical tolerances from the operation, dtype and oracle; justify them before optimizing. Choose task-quality and latency budgets after observing the baseline and before selecting an optimization. Include edge cases, repetitions, raw data and failure counts. There is no universal required percentage speedup or token rate.

If the apparent improvement is smaller than run-to-run variation, report it as inconclusive. If a smaller representation is slower, explain conversion, dispatch or memory costs. An optimization that loses is still useful evidence if the method and diagnosis are sound.

## Final assessment

Complete the week-39 reconstruction tasks without consulting the old implementation first. Show a colleague or use a recorded self-review to test whether the README is sufficient. Verify that every portfolio claim points to an artifact and every unfinished feature is labeled. Your next step should address one measured gap, not repeat the full reading list.
