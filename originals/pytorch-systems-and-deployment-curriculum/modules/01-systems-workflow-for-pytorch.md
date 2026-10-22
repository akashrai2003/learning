# Module 01 — Systems Workflow for PyTorch

> **Phase:** I — Engineering foundations  
> **Time:** Week 1; three focused 3-hour sessions  
> **Prerequisites:** Comfortable Python, Git, virtual environments, and elementary linear algebra

## Existing inference curriculum bridge

Read [Module 01 — `inference-system-map.md`](../../inference-engineering-curriculum/modules/01-inference-system-map.md), [Module 02 — `reproducible-stack-and-toolchain.md`](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), and [Module 44 — `profiling-benchmarking-and-correctness.md`](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). That curriculum begins from a request-serving system; this one begins from a small PyTorch codebase whose tensors, state, tests, artifacts, and measurements remain intelligible as it grows into that system.

## Why this matters and outcomes

PyTorch proficiency is not memorising `nn.Linear` calls. It is being able to turn a numerical claim into a reviewable program, say which state changes on each step, and reproduce a performance or accuracy result six weeks later. A notebook is excellent for an idea; it is a poor boundary for a compiler experiment, model artifact, or upstream bug report. Build one durable repository, called **`torch-lab/`**, throughout this course.

By the end you can create a runnable package, state its tensor and device contracts, separate configuration from code, write a fast smoke test, record environment provenance, and distinguish a unit test, numerical oracle, integration test, and benchmark. You will also have a contribution-ready issue template: minimal reproduction, expected versus actual behavior, versions, and a test that would prevent regression.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P01 — Systems workflow for PyTorch**.

1. [PyTorch contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) — learn the API or system contract.
2. [PyTorch developer notes](https://docs.pytorch.org/docs/stable/community/contribution_guide.html) — study the underlying design, paper, or specification.
3. [PyTorch benchmarks](https://github.com/pytorch/pytorch/tree/main/benchmarks) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch developer notes](https://docs.pytorch.org/docs/stable/community/contribution_guide.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch source repository](https://github.com/pytorch/pytorch) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch developer discussion](https://dev-discuss.pytorch.org/) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model: an experiment is a state transition system

Treat a run as a pure-ish function plus explicitly named state:

\[
(\theta_t, o_t, r_t, c, x_t) \longmapsto (y_t, \theta_{t+1}, o_{t+1}, r_{t+1}, a_t).
\]

Here \(\theta\) is model state, \(o\) optimizer state, \(r\) RNG state, \(c\) immutable configuration, \(x\) input, \(y\) output, and \(a\) artifacts/metrics. In inference, \(o\) is absent but model weights, tokenizer revision, KV cache, scheduler policy, and compiled-kernel cache are still state. If a result cannot name its inputs, executable environment, and state transitions, it is an observation rather than evidence.

Make boundaries explicit:

```text
CLI/config -> data adapter -> model -> loss/decode -> metrics/artifacts
                        |                   |
                    state_dict          benchmark harness
```

The model should accept tensors and return tensors or a typed result; filesystem, logging, HTTP, and command-line parsing belong outside it. That boundary makes eager parity, `torch.compile`, export, and a custom operator independently testable later.

## Theory, tensor contracts, and code organization

Adopt a dimension vocabulary early: `B` batch, `T` tokens/time, `D` hidden width, `V` vocabulary, `H` heads, and `Dh=D/H`. A decoder contract might say `input_ids: int64[B,T]`, `logits: float32-or-low-precision[B,T,V]`, and loss scalar `float32[]`. A contract must also state device, dtype, layout expectation, mutation/aliasing behavior, and whether gradients are expected. Shape alone is insufficient: a `[B,T]` tensor could be token IDs, mask, or labels.

Proposed repository layout—create it gradually, not as empty ceremony:

```text
torch-lab/
  pyproject.toml          src/torch_lab/{config,models,ops,training,serve}/
  tests/{unit,numerics,integration}/  benchmarks/  scripts/  docs/runs/
  artifacts/              # ignored; manifest and hashes are committed
```

Use a single entry point such as `python -m torch_lab.train --config ...`; a config file is data, not hidden global state. Every run writes resolved configuration, Git commit/dirty status, PyTorch/CUDA versions, device name, seed, model/data hashes, and timing protocol. Do not commit weights, caches, tokens, private data, or generated binaries.

## Implementation ladder in the proposed `torch-lab/`

1. **Reference:** `src/torch_lab/models/toy_mlp.py` implements a deterministic two-layer classifier with plain tensor operations and an explicit `forward(x) -> logits` contract. Test shape, device, and a hand-computed two-example output.
2. **Optimized:** add a device/dtype policy, `torch.inference_mode()` evaluation path, and a benchmark function using CUDA events when applicable. Preserve the reference output within a declared tolerance; do not optimise a changing oracle.
3. **Integrated:** expose `train`, `eval`, and `benchmark` CLI commands using the same model factory and artifact manifest. Add a tiny CPU CI run and a GPU-optional test marker. This becomes the harness used by every later module.

## Three focused sessions

1. **Session A — boundaries (3h):** initialise the package; define `RunConfig` with dataclasses; implement the toy model; annotate every input/output with shape, dtype, and device; write one unit test before adding a training loop.
2. **Session B — evidence (3h):** add structured JSONL metrics, a `collect_env.py` script, deterministic synthetic data, and a numerical test that compares against a literal Python/NumPy calculation. Intentionally change a seed and verify which artifacts change.
3. **Session C — operation (3h):** make command execution, failure messages, and test selection ergonomic. Time a warm CPU run and, if available, a GPU run; record that the first GPU call includes initialization and must not be compared to warm iterations.

## Experiments and deliverables

Commit a `docs/runs/0001-toy-baseline.md` containing the hypothesis, command, manifest, tensor contracts, raw timing samples, and interpretation. Add tests that reject a wrong token dtype, mismatched device, and accidental output mutation. Make a deliberately malformed config fail before allocating CUDA memory. These small failures teach more than a successful demo.

## Common failure modes

- Hiding `device`, dtype, seed, or model revision in module globals, so two commands with identical flags are not equivalent.
- Calling `.item()`, printing CUDA tensors, or synchronizing inside a timed loop and reporting enqueue time as execution time.
- Storing a live module object with arbitrary Python state instead of a documented artifact contract.
- Mixing data download, preprocessing, model definition, and benchmarking in one script, which makes an upstream minimal reproduction impossible.
- Treating a passing stochastic text sample as numerical correctness.

## Mastery gate

From a clean clone and one documented command, reproduce a CPU result and a GPU-optional result, produce a manifest, and make all tests pass. In a code review, explain every mutable state component and show a test that catches a change to it. Submit a small documentation or test-only pull request to one PyTorch-adjacent repository after reading its contribution rules; the goal is learning review mechanics, not premature kernel work.

## Primary sources and canonical code paths

- [PyTorch developer notes](https://docs.pytorch.org/docs/stable/community/contribution_guide.html) and [PyTorch contributing guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md).
- [PyTorch testing utilities](https://docs.pytorch.org/docs/stable/testing.html) and the repository's [`test/`](https://github.com/pytorch/pytorch/tree/main/test) tree.
- Inspect [`torch/nn/modules/module.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py), [`torch/utils/benchmark`](https://github.com/pytorch/pytorch/tree/main/torch/utils/benchmark), and [`torch/utils/collect_env.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/collect_env.py).

## OSS bridge and next

Good first contributions are isolated test improvements, docs fixes, or minimal regressions with an executable reproducer. Learn labels and CI locally before touching dispatcher or CUDA code. Next, make the Python and native dependency graph itself reproducible in [Module 02](02-reproducible-python-cuda-and-packaging.md).
