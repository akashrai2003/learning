# Concept Map and Dependency Order

The course deliberately moves from **semantic correctness** to **graph and
native correctness**, then to **system and artifact correctness**. Skipping
left-to-right dependencies is how one ends up with a fast benchmark that has no
defined model contract.

```text
Reproducible systems + numerical discipline (01–06)
        │
        ├── Tensor semantics + autograd + state (07–16)
        │       │
        │       ├── Functional transforms + profiling (17–18)
        │       └── Training engine + Transformer + KV cache (19–21)
        │                   │
        ├── AMP / tuning / TorchAO / modalities / evaluation (22–28)
        │                   │
        └── FX → Dynamo → AOTAutograd → Inductor → Triton (29–34)
                            │                 │
                            ├── Custom-op contract (35)
                            │         │
                            │         └── C++/CUDA → dispatcher/codegen → ABI/build (36–40)
                            │                              │
                            └── CUDA Graphs + compiler debugging (41–42)
                                                              │
                         Process groups → DDP/FSDP/DTensor → parallelism/reliability (43–50)
                                                              │
                     torch.export → ONNX/ORT | TensorRT | AOTInductor | ExecuTorch (51–58)
                                                              │
                           Source build → CI/triage → upstream-quality contribution (59–62)
```

## The four correctness layers

| Layer | Question | Typical oracle | Typical failure |
|---|---|---|---|
| Tensor/numerical | Does the operator compute the intended value? | Eager reference, gradcheck, property test | Wrong view, dtype, reduction, mask, gradient |
| Graph/transform | Is the program valid under capture, compile, export, and dynamic shapes? | FX/ExportedProgram inspection, `opcheck`, compile traces | Graph break, guard churn, alias/mutation violation, missing fake behavior |
| Native/runtime | Does the generated or extension implementation load and execute correctly? | C++/CUDA parity tests, build logs, clean environment | ABI/linker failure, architecture mismatch, host OOM, invalid stream/lifetime |
| System/artifact | Does it satisfy production placement, reliability, and target constraints? | Distributed/target-runtime tests, deployment dossier | Mis-sharding, collective hang, delegate fallback, unsupported artifact/input |

Every optimized path must preserve all layers below it. A TensorRT engine that
runs but changes outputs has failed layer 1. A CUDA extension that is correct in
eager mode but cannot participate in `torch.compile` has failed layer 2. A
compiled path that runs only with a stale cached library has failed layer 3.

## Parallel learning lanes

Once you reach Module 20, keep two lanes alive:

| Primary lane | Secondary lane | Reason |
|---|---|---|
| A compact Transformer/LLM artifact | PyTorch source walk | Every abstraction is tied to a real workload and production code |
| One kernel/custom-op problem | Performance/reproducibility ledger | Prevents tuning unsupported or incorrect cases |
| One export target | Eager parity test suite | Makes deployment failure diagnosis localizable |
| One contribution candidate | Project tests/CI conventions | Turns study into upstream-relevant practice |

For your interests, the ideal mid-course loop is:

```text
KV cache / low-bit Transformer workload
       → profile in eager PyTorch
       → inspect compile graph
       → implement a narrow custom operator
       → validate compile/export behavior
       → compare a serving/runtime integration
       → write a source walk or focused issue/patch
```

This is the shortest path from “I can use PyTorch” to “I can explain and change
the performance stack responsibly.”
