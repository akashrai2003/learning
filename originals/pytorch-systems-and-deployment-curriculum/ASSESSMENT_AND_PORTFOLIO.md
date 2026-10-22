# Assessment and Portfolio Rubric

The course becomes career-relevant when its evidence can survive independent
review. Use this rubric after each phase and at the end of every project.

| Dimension | Developing | Strong | Contribution-ready |
|---|---|---|---|
| Semantic reasoning | Can use APIs | Predicts shapes/dtypes/state and writes edge tests | Identifies contracts across eager/autograd/dispatch/transforms |
| Numerical work | Notices a mismatch | Explains tolerance and stability causes | Designs precision/quantization experiments with valid quality evidence |
| Performance work | Reports a timing | Has a benchmark protocol and bottleneck hypothesis | Connects traces, roofline/resource model, implementation, and regression test |
| Compiler/native work | Runs `torch.compile` or an extension | Diagnoses graph breaks/build failures | Defines/registers/tests an op across relevant transforms and toolchains |
| Distributed/deployment | Uses a framework flag | Explains placement/artifact constraints | Produces recovery/portability evidence and clear scope limits |
| OSS practice | Reads source | Writes source walks and local reproducers | Proposes/reviews a narrow, tested, maintainable upstream change |

## Portfolio minimum at the 3-, 6-, and 12-month marks

| Milestone | Evidence to retain |
|---|---|
| 3 months | Tensor/autograd microscope, minimal training engine, Transformer/KV-cache reference, baseline profiler reports, 4–6 source walks |
| 6 months | AMP/TorchAO evidence, one torchtune modification, a compiler/custom-op dossier, native-build postmortem, one small issue/reproducer or documentation contribution |
| 9–12 months | Distributed dossier, validated export/runtime artifact, upstream-ready patch, six capstone READMEs, and a clear specialization narrative |

## Interview / design-review prompts to practice

- Why can a tensor operation be correct yet slow after a transpose or view?
- What makes a custom operator compatible with FakeTensor, autograd,
  `torch.compile`, and export?
- How do guard failures and dynamic-shape assumptions affect a production
  workload with variable sequence length?
- Why is a lower-bit checkpoint not proof of higher generation throughput?
- How do DDP, FSDP, DTensor, tensor parallelism, and serving parallelism differ
  in ownership and communication?
- What is the executable contract of an ONNX, TensorRT, AOTInductor, or
  ExecuTorch artifact?
- How would you reduce a Blackwell native-build failure before asking an
  upstream maintainer for help?

If you can answer these with a concrete experiment and code artifact—not just
definitions—you have reached the practical depth this curriculum targets.
