# Module 15 — Randomness, Determinism, and Reproducibility

> **Phase:** II — Reliable PyTorch systems  
> **Time:** 8–10 focused hours across three sessions  
> **Prerequisites:** Modules 01–14 and a working data/optimizer loop

## Existing inference curriculum bridge

Connect training reproducibility to [Module 02 — Reproducible Stack and Toolchain](../../inference-engineering-curriculum/modules/02-reproducible-stack-and-toolchain.md), stochastic decoding in [Module 32 — Decoding, Sampling, and Logits Processing](../../inference-engineering-curriculum/modules/32-decoding-sampling-and-logits-processing.md), and measurement discipline in [Module 44 — Profiling, Benchmarking, and Correctness](../../inference-engineering-curriculum/modules/44-profiling-benchmarking-and-correctness.md). A seed is only one component of a reproducible experiment or serving request.

## Why this matters and outcomes

“Set the seed” is not a reproducibility strategy. Initialization, dropout, sampling, augmentation, Dataset workers, Python/NumPy libraries, kernel algorithms, process order, and hardware all influence results. The useful goal is a named reproducibility level, not a false promise of bitwise equality across releases and platforms.

By the end, you can inventory RNG and nondeterminism sources; construct deterministic unit tests and bounded-tolerance integration tests; preserve enough state for faithful resume; configure deterministic algorithms intentionally; and explain why concurrent generation needs request-scoped sampling state rather than accidental global order.

## Guided study sources

Follow this sequence alongside the module’s implementation ladder. The full
four-source progression, including the production source to inspect, is in the
[PyTorch systems source atlas](../resources/SOURCE_ATLAS.md) under **P15 — Randomness, determinism, and reproducibility**.

1. [Reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html) — learn the API or system contract.
2. [PyTorch random API](https://docs.pytorch.org/docs/stable/random.html) — study the underlying design, paper, or specification.
3. [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) — implement, profile, or validate the behavior.

Use the existing source section below as supplemental reading; the ordered path
above tells you what to learn and build first.

### Further study and source walks

- [PyTorch examples](https://github.com/pytorch/examples) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.
- [PyTorch recipes](https://docs.pytorch.org/tutorials/recipes/recipes_index.html) — read this after the core sequence; extract one idea, code path, or experiment for your notes.

## Mental model and theory

An RNG state plus a sequence of draws determines a sequence of pseudo-random values. It does not determine values if an unseen source draws from another generator, if worker scheduling changes draw order, or if an operator itself is nondeterministic. PyTorch has CPU and accelerator RNG state, and a torch.Generator can make a source explicit. Python random and NumPy generator objects are independent. A DataLoader worker is a separate process with separate state.

Define the target before configuring flags:

- **Unit reproducibility:** a fixed, tiny input yields exact output under one installed build.
- **Run reproducibility:** the same configuration, data ordering, hardware, and software yields the same training trajectory or a declared tolerance.
- **Resume reproducibility:** a restart at step t continues the same state transition sequence.
- **Cross-environment comparability:** results are scientifically comparable, usually with tolerance and distribution checks rather than bitwise identity.

torch.manual_seed establishes PyTorch seed state, but a full process typically also seeds Python and NumPy. Use explicit Generators where ownership matters. For worker loading, pass a DataLoader generator and a worker initializer that derives Python/NumPy seeds from torch.initial_seed. In an IterableDataset, deterministic worker seeding does not repair duplicated or changing shard assignment; Module 13’s sample-ID checks remain necessary.

torch.use_deterministic_algorithms(True) asks PyTorch to use deterministic alternatives where available and raise for known unsupported nondeterministic operations. It can be slower, use more memory, or block an otherwise valid kernel. Backend autotuning and floating-point reductions add further variation. The official PyTorch note explicitly does not promise reproducibility across releases, commits, platforms, or CPU versus GPU. Treat a deterministic failure as useful capability evidence, not permission to silently disable the setting.

For generation, sampling draws must be associated with a request and model/tokenizer configuration. With concurrent batches, a global generator’s draw order may change with scheduler decisions; “same seed” then fails to define the same request outcome. Define seed, generator/algorithm, prompt, decoding configuration, model revision, and any request ordering policy in the trace.

## Reference → optimized → integrated ladder in proposed torch-lab

**Reference**

1. Create torch-lab/repro/rng_inventory.py. Seed Python, NumPy, PyTorch CPU, and CUDA; record random draws, generator states, and the order in which each source is consumed.
2. Make a tiny dropout-and-augmentation training example that is exact in a pinned local environment and prove which omitted seed changes it.

**Optimized**

3. Add a multi-worker DataLoader with a passed generator and worker initializer. Test IDs, augmentation values, and batches for repeated runs, then intentionally alter worker count and explain the changed scope.
4. Enable deterministic algorithms in a focused test suite. Record operations that error, performance/memory changes, and the minimum configuration that reproduces a difference.

**Integrated**

5. Add a run manifest containing seed policy, RNG snapshots, Python/NumPy/PyTorch/CUDA versions, device, deterministic settings, data revision, and logical step. Restore it into a resume-equivalence test.
6. Implement a toy request sampler with a per-request generator and demonstrate why a global generator is scheduling-order dependent.

## Three focused sessions

1. **RNG accounting (3h):** map every draw in a small run, save/restore generator state, and distinguish seed from generator state.
2. **Data and kernels (3h):** add workers, test sharding and seed behavior, then enable deterministic algorithms and capture one expected limitation.
3. **Resume and serving (3h):** build an exact local resume test and a request-scoped sampling test under different simulated batch orders.

## Experiments and deliverables

Commit an RNG inventory, deterministic-policy document, worker reproducibility test, run manifest schema, and two experiment reports: exact same-environment replay and deliberately changed-environment comparison. State whether each claim is exact, tolerant, or statistical. Never report “reproducible” without scope.

## Common failure modes

- Seeding only torch while Python, NumPy, tokenizer, or workers remain uncontrolled.
- Assuming a seed resets an already-consumed generator.
- Treating deterministic algorithms as cross-version bitwise guarantees.
- Letting concurrent requests share a global sampling stream.
- Disabling determinism after an error without recording the affected operation.
- Saving model weights but not RNG, sampler, step, or schedule state.

## Mastery gate

Reproduce a short local run exactly after restart, then explain precisely why it may differ after a version, device, worker-count, or request-scheduling change. Demonstrate two concurrent sampling orders where global state differs and request-scoped state remains defined.

## Primary sources and canonical code paths

- [PyTorch reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch deterministic algorithms](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
- [PyTorch Generator API](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html)
- [PyTorch DataLoader API](https://docs.pytorch.org/docs/stable/data.html)
- Inspect [GeneratorImpl.h](https://github.com/pytorch/pytorch/blob/main/c10/core/GeneratorImpl.h), [Generator.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Generator.cpp), and DataLoader worker code.

## OSS bridge and next

Useful upstream work includes a minimal deterministic-algorithm regression, a worker seeding test, or documentation that defines a reproducibility boundary accurately. Avoid a broad “nondeterministic” report without versions, inputs, and a repeatable comparison. Module 16 serializes the state inventory safely and makes it a model artifact contract.
